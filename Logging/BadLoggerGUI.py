import sys
from itertools import chain

from DataLoading.Trigger import Operation
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem

from DataLoading import Trigger
from Logging.Logger import warn_log_collection, debug_log_collection, info_log_collection, error_log_collection

"""
    This logger GUI is bad. You can use it, but you will have to manually shut it down, plus there
    are are quite a few bugs. Someone who knows more about pyqt should add events/signals etc.
    to make it proper. It's also severly incomplete and there is no filtering yet.
    
    Feel free to make it better :-)
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)

    listWidget = QTableWidget()
    listWidget.setAlternatingRowColors(True)
    listWidget.setColumnCount(4)
    listWidget.setColumnWidth(0, 250)
    listWidget.setColumnWidth(1, 80)
    listWidget.setColumnWidth(2, 530)
    listWidget.setColumnWidth(3, 200)

    listWidget.setFont(QFont('Courier New', 10))
    listWidget.setHorizontalHeaderLabels(['Date', 'Severity', 'Message', 'Caller'])
    listWidget.resize(1100, 600)
    listWidget.move(200, 100)
    listWidget.setWindowTitle('Photon Log-Analyser')

    listWidget.show()


    def add_doc_to_list(doc):
        rowPosition = 0
        listWidget.insertRow(rowPosition)
        listWidget.setItem(rowPosition, 0, QTableWidgetItem(doc['logged_date'].strftime('%d %b %Y - %H:%M:%S.%f')))
        listWidget.setItem(rowPosition, 1, QTableWidgetItem(doc['log_type']))
        listWidget.setItem(rowPosition, 2, QTableWidgetItem(doc['message']))
        listWidget.setItem(rowPosition, 3, QTableWidgetItem(doc['called_by']))


    """ Initial loading of all logs. Should later be capped to newest XYZ documents from each collection """

    debug = debug_log_collection.find()
    info = info_log_collection.find()
    warn = warn_log_collection.find()
    error = error_log_collection.find()

    data = [x for x in chain(debug, info, warn, error)]

    sortedArray = sorted(
        data,
        key=lambda x: x['logged_date'], reverse=False
    )

    for w in sortedArray:
        add_doc_to_list(w)

    """ Observer and add new logs from oplog """

    def add_log(oplog_doc):
        doc = oplog_doc["o"]
        add_doc_to_list(doc)


    Trigger.subscribe_to_collection('photon_manager_db.debug_log', add_log, Operation.INSERT)
    Trigger.subscribe_to_collection('photon_manager_db.info_log', add_log, Operation.INSERT)
    Trigger.subscribe_to_collection('photon_manager_db.warn_log', add_log, Operation.INSERT)
    Trigger.subscribe_to_collection('photon_manager_db.error_log', add_log, Operation.INSERT)

    sys.exit(app.exec_())
