import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import uic
from threading import Thread, Event
import logging
import time
import schedule
import gunshot  # This imports the functions and constants from gunshot.py

# Set up logger to view logs in the UI
logger = logging.getLogger('debugger')
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('output.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Global variables to control the audio detection threads
status_stop_event = Event()
detection_thread = None

class GunshotApp(QMainWindow):
    def __init__(self):
        super(GunshotApp, self).__init__()
        uic.loadUi('gunshot_detection.ui', self)

        # Connect UI buttons to functions
        self.startButton.clicked.connect(self.start_detection_process)
        self.stopButton.clicked.connect(self.stop_detection_process)
        self.logButton.clicked.connect(self.show_recent_log)

        # Initial status for the status label
        self.statusLabel.setText("Status: Not running")

        # Track the running state of the detection process
        self.detection_running = False

    def start_detection_process(self):
        global detection_thread
        if not self.detection_running:
            logger.debug("Starting gunshot detection")
            self.statusLabel.setText("Status: Running")
            self.detection_running = True

            # Start the detection thread
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

        # Start SMS alert thread (daemon)
        sms_alert_thread = Thread(target=gunshot.send_sms_alert, daemon=True)
        sms_alert_thread.start()

        # Start status monitoring thread (daemon)
        gunshot.status_thread = Thread(target=gunshot.print_status, args=(status_stop_event,), daemon=True)
        gunshot.status_thread.start()

        # Initialize the PyAudio stream from gunshot.py
        gunshot.stream.start_stream()
        logger.debug("Listening to audio stream...")

        # Run detection process (based on gunshot.py)
        try:
            while not status_stop_event.is_set():
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            status_stop_event.set()

        # Stop the stream and threads
        gunshot.stream.stop_stream()
        gunshot.stream.close()
        logger.debug("Stopped listening to audio stream")

    def show_recent_log(self):
        # Only show the last few lines of the log to avoid overwhelming the UI
        log_file = 'output.log'
        try:
            with open(log_file, 'r') as file:
                log_data = file.readlines()
                recent_logs = log_data[-10:]  # Only take the last 10 lines
                log_output = ''.join(recent_logs)
        except FileNotFoundError:
            log_output = "Log file not found."

        # Display the log output in a message box
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Recent Log")
        msg_box.setText(log_output)
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GunshotApp()
    window.show()
    sys.exit(app.exec_())
