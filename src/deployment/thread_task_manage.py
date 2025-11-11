import task
import argparse
import os
from threading import Thread, Semaphore, Event
import utils
import logging
from logging import FileHandler
import pandas as pd
import time
import inspect

class InstantFlushFileHandler(FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# 配置日志格式和文件
logging.basicConfig(
    level=logging.INFO,  # 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='./logs/landslide_task.log',  # 日志文件路径
    filemode='a'  # 追加模式（'w'为覆盖） 
)
logger = logging.getLogger(__name__)
logger.addHandler(InstantFlushFileHandler('./logs/landslide_task.log'))

semaphore = Semaphore(1)

def _start_task_thread(serial_id, device_id, height_width, semaphore, logger, event_type, videoid,
                       mqtt_host, mqtt_port, mqtt_username, mqtt_password, stop_event, enabled):
    """
    wrapper that will call task.landslide_task. If the landslide_task supports a stop_event kwarg
    it will be passed, otherwise the task is invoked as-is (cannot be stopped externally).
    """
    try:
        sig = inspect.signature(task.landslide_task)
        # prepare positional args used earlier: (serial_id, device_id, (height, width), 1, 20, semaphore, logger, event_type, videoid, mqtt_host, mqtt_port, mqtt_username, mqtt_password)
        base_args = (serial_id, device_id, height_width, 1, 20, semaphore, logger, event_type, videoid,
                     mqtt_host, mqtt_port, mqtt_username, mqtt_password, enabled)
        if 'stop_event' in sig.parameters:
            task.landslide_task(*base_args, stop_event=stop_event)
        else:
            # landslide_task doesn't accept stop_event: run normally (cannot be force-stopped)
            task.landslide_task(*base_args)
    except Exception as e:
        logger.exception("Exception in landslide task for %s: %s", serial_id, e)


def main(args):
    mask_path = args.mask_path
    height = args.height
    width = args.width
    poll_interval = args.poll_interval

    # mqtt config
    mqtt_host = os.environ.get("MQTT_HOST", "broker.emqx.io")
    mqtt_port = int(os.environ.get("MQTT_PORT", 1883))
    mqtt_username = os.environ.get("MQTT_USERNAME", None)
    mqtt_password = os.environ.get("MQTT_PASSWORD", None)
    # get csv data
    mapping_csv_path = os.environ.get("MAPPING_CSV_PATH", "/app/data/mapping.csv")

    camera2ctrl = {}  # serial_id -> {'thread': Thread, 'stop_event': Event, 'device_id': str}

    def load_mapping():
        mapping = pd.read_csv(mapping_csv_path)
        return mapping.set_index("设备序列号")['视频ID'].to_dict()

    # initial load + start
    serial2videoid = load_mapping()
    camera_info_list = utils.get_camera_info_list() # (serial_no, device_id)
    hk_service_camera_index_code_list = [elem[1] for elem in camera_info_list]

    def refresh_and_apply():
        nonlocal serial2videoid, camera2ctrl
        serial2videoid = load_mapping()
        # print(f"serial2video: {serial2videoid}")s
        camera_info_list_from_db = utils.get_camera_info_list_from_database() # (..., device_id, serial_no, ...)
        # logger.info(f"camera list from db: {camera_info_list_from_db}")
        desired_serials = set()
        desired_info = {}
        for camera_info in camera_info_list_from_db:
            device_id = camera_info['device_id']
            serial_id = camera_info['serial_no']
            enabled = camera_info['enabled']
            if device_id not in hk_service_camera_index_code_list:
                continue
            if serial_id not in serial2videoid.keys():
                continue
            if camera_info.get('llm_model_type', '') == '':
                continue
            desired_serials.add(serial_id)
            desired_info[serial_id] = {
                'device_id': device_id,
                'event_type': camera_info['llm_model_type'],
                'videoid': str(serial2videoid[serial_id]),
                'enabled': enabled
            }
        # logger.info(desired_info)
        # start new tasks
        for serial in desired_serials - set(camera2ctrl.keys()):
            info = desired_info[serial]
            stop_event = Event()
            th = Thread(
                target=_start_task_thread,
                args=(serial, info['device_id'], (height, width), semaphore, logger,
                      info['event_type'], info['videoid'],
                      mqtt_host, mqtt_port, mqtt_username, mqtt_password, stop_event, info['enabled']),
                daemon=False
            )
            th.start()
            camera2ctrl[serial] = {'thread': th, 'stop_event': stop_event, 'device_id': info['device_id'], 'enabled': info['enabled']}
            logger.info("Started task for %s (device %s) (enabled %s)", serial, info['device_id'], info['enabled'])

        # handle enabled flag changes: restart thread when enabled value toggles
        for serial in set(camera2ctrl.keys()) & desired_serials:
            current_enabled = camera2ctrl[serial].get('enabled', True)
            desired_enabled = desired_info[serial]['enabled']
            if current_enabled != desired_enabled:
                logger.info("Enabled changed for %s: %s -> %s. Restarting thread.", serial, current_enabled, desired_enabled)
                ctrl = camera2ctrl[serial]
                ctrl['stop_event'].set()
                ctrl['thread'].join(timeout=5)
                if ctrl['thread'].is_alive():
                    logger.warning("Previous task thread for %s did not exit; starting replacement anyway.", serial)
                # remove old entry
                camera2ctrl.pop(serial, None)
                # start new thread with updated enabled
                info = desired_info[serial]
                stop_event = Event()
                th = Thread(
                    target=_start_task_thread,
                    args=(serial, info['device_id'], (height, width), semaphore, logger,
                          info['event_type'], info['videoid'],
                          mqtt_host, mqtt_port, mqtt_username, mqtt_password, stop_event, info['enabled']),
                    daemon=False
                )
                th.start()
                camera2ctrl[serial] = {'thread': th, 'stop_event': stop_event, 'device_id': info['device_id'], 'enabled': info['enabled']}
                logger.info("Restarted task for %s (device %s) (enabled %s)", serial, info['device_id'], info['enabled'])

        # stop removed tasks
        for serial in set(camera2ctrl.keys()) - desired_serials:
            ctrl = camera2ctrl[serial]
            logger.info("Marking task to stop for %s", serial)
            # Signal stop; actual stop requires landslide_task to accept and honor stop_event.
            ctrl['stop_event'].set()
            # try to join for a short time; if thread doesn't exit, leave it and log
            ctrl['thread'].join(timeout=5)
            if ctrl['thread'].is_alive():
                logger.warning("Task thread for %s did not exit after stop_event; it may not support graceful stop.", serial)
            else:
                logger.info("Task thread for %s stopped cleanly.", serial)
                camera2ctrl.pop(serial, None)

    # initial apply
    logger.info("Initial camera list: %s", camera_info_list)
    refresh_and_apply()

    try:
        while True:
            time.sleep(poll_interval)
            logger.info("Refreshing camera config from database...")
            refresh_and_apply()
    except KeyboardInterrupt:
        logger.info("Shutdown requested, signaling all tasks to stop.")
        for serial, ctrl in list(camera2ctrl.items()):
            ctrl['stop_event'].set()
            ctrl['thread'].join(timeout=5)
        logger.info("Exiting main loop.")


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-m', '--mask_path', type=str, default="./export_masks",
                   help=('mask path'))
    p.add_argument('-o', '--output_dir', type=str, default="./outputs",
                   help='Output directory where results will be saved.')
    p.add_argument('-l', '--length', type=int, default=10,
                   help=('history window length'))
    p.add_argument('--height', type=int, default=640)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--show', type=bool, default=True)
    p.add_argument('--poll-interval', dest='poll_interval', type=int, default=60,
                   help='Seconds between refreshing DB camera configuration.')

    main(p.parse_args())