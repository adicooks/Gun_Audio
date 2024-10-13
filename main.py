import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import uic
from threading import Thread, Event
import logging
import time
import schedule
import gunshot

logger = logging.getLogger('debugger')
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('output.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

status_stop_event = Event()
detection_thread = None

class GunshotApp(QMainWindow):
    def __init__(self):
        super(GunshotApp, self).__init__()
        uic.loadUi('gunshot_detection.ui', self)

        self.startButton.clicked.connect(self.start_detection_process)
        self.stopButton.clicked.connect(self.stop_detection_process)
        self.logButton.clicked.connect(self.show_recent_log)

        self.statusLabel.setText("Status: Not running")

        self.detection_running = False

    def start_detection_process(self):
        global detection_thread
        if not self.detection_running:
            logger.debug("Starting gunshot detection")
            self.statusLabel.setText("Status: Running")
            self.detection_running = True

            detection_thread = Thread(target=self.run_detection, daemon=True)
            detection_thread.start()
        else:
            logger.debug("Detection process already running")
            self.statusLabel.setText("Status: Already running")

    def stop_detection_process(self):
        global status_stop_event
        if self.detection_running:
            logger.debug("Stopping gunshot detection")
            status_stop_event.set()
            self.statusLabel.setText("Status: Stopped")
            self.detection_running = False
        else:
            logger.debug("No active detection process to stop")
            self.statusLabel.setText("Status: Already stopped")

    def run_detection(self):
        global status_stop_event
        status_stop_event.clear()

        sms_alert_thread = Thread(target=gunshot.send_sms_alert, daemon=True)
        sms_alert_thread.start()

        gunshot.status_thread = Thread(target=gunshot.print_status, args=(status_stop_event,), daemon=True)
        gunshot.status_thread.start()

        gunshot.stream.start_stream()
        logger.debug("Listening to audio stream...")

        try:
            while not status_stop_event.is_set():
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            status_stop_event.set()

        gunshot.stream.stop_stream()
        gunshot.stream.close()
        logger.debug("Stopped listening to audio stream")

    def show_recent_log(self):
        log_file = 'output.log'
        try:
            with open(log_file, 'r') as file:
                log_data = file.readlines()
                recent_logs = log_data[-10:]
                log_output = ''.join(recent_logs)
        except FileNotFoundError:
            log_output = "Log file not found."

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Recent Log")
        msg_box.setText(log_output)
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GunshotApp()
    window.show()
    sys.exit(app.exec_())
