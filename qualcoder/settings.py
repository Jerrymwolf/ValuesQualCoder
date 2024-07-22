# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Colin Curtain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Author: Colin Curtain (ccbogel)
https://github.com/ccbogel/QualCoder
https://qualcoder.wordpress.com/
"""

from PyQt6 import QtGui, QtWidgets, QtCore
import os
import sys
import logging
import traceback
import copy

from .GUI.ui_dialog_settings import Ui_Dialog_settings
from .helpers import Message

home = os.path.expanduser('~')
path = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


def exception_handler(exception_type, value, tb_obj):
    """ Global exception handler useful in GUIs.
    tb_obj: exception.__traceback__ """
    tb = '\n'.join(traceback.format_tb(tb_obj))
    text = 'Traceback (most recent call last):\n' + tb + '\n' + exception_type.__name__ + ': ' + str(value)
    print(text)
    logger.error(_("Uncaught exception: ") + text)
    if len(text) > 500:
        text = _('Shortened error message: ...') + text[-500:]
    QtWidgets.QMessageBox.critical(None, _('Uncaught Exception'), text)


class DialogSettings(QtWidgets.QDialog):
    """ Settings for the coder name, coder table and to display ids. """

    settings = {}
    current_coder = "default"

    def __init__(self, app, parent=None, section=None, enable_ai=False):

        sys.excepthook = exception_handler
        self.app = app
        self.settings = app.settings
        if enable_ai:
            self.settings['ai_enable'] = 'True'
        self.ai_models = copy.deepcopy(self.app.ai_models)
        self.current_coder = self.app.settings['codername']
        super(QtWidgets.QDialog, self).__init__(parent)  # overrride accept method
        QtWidgets.QDialog.__init__(self)
        self.ui = Ui_Dialog_settings()
        self.ui.setupUi(self)
        font = f"font: {self.app.settings['fontsize']}pt "
        font += '"' + self.app.settings['font'] + '";'
        self.setStyleSheet(font)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        new_font = QtGui.QFont(self.settings['font'], self.settings['fontsize'], QtGui.QFont.Weight.Normal)
        self.ui.label_current_coder.setText(_("Current coder: ") + self.app.settings['codername'])
        self.ui.fontComboBox.setCurrentFont(new_font)
        # Get coder names from all tables
        sql = "select owner from  code_image union select owner from code_text union select owner from code_av "
        sql += " union select owner from cases union select owner from journal union select owner from attribute "
        sql += "union select owner from source union select owner from annotation union select owner from code_name "
        sql += "union select owner from code_cat"
        coders = [""]
        if self.app.conn is not None:
            cur = self.app.conn.cursor()
            cur.execute(sql)
            results = cur.fetchall()
            for row in results:
                if row[0] != "":
                    coders.append(row[0])
        self.ui.comboBox_coders.addItems(coders)
        languages = ["Deutsch de", "English en", "Español es", "Français fr",
                     "Italiano it", "Português, pt"]
        self.ui.comboBox_language.addItems(languages)
        for index, lang in enumerate(languages):
            if lang[-2:] == self.settings['language']:
                self.ui.comboBox_language.setCurrentIndex(index)
        timestampformats = ["[mm.ss]", "[mm:ss]", "[hh.mm.ss]", "[hh:mm:ss]",
                            "{hh:mm:ss}", "#hh:mm:ss.sss#"]
        self.ui.comboBox_timestamp.addItems(timestampformats)
        for index, ts in enumerate(timestampformats):
            if ts == self.settings['timestampformat']:
                self.ui.comboBox_timestamp.setCurrentIndex(index)
        speakernameformats = ["[]", "{}"]
        self.ui.comboBox_speaker.addItems(speakernameformats)
        for index, snf in enumerate(speakernameformats):
            if snf == self.settings['speakernameformat']:
                self.ui.comboBox_speaker.setCurrentIndex(index)
        self.ui.spinBox.setValue(self.settings['fontsize'])
        self.ui.spinBox_treefontsize.setValue(self.settings['treefontsize'])
        self.ui.spinBox_docfontsize.setValue(self.settings['docfontsize'])
        self.ui.comboBox_coders.currentIndexChanged.connect(self.combobox_coder_changed)
        index = self.ui.comboBox_text_chunk_size.findText(str(self.settings['codetext_chunksize']),
                                                          QtCore.Qt.MatchFlag.MatchFixedString)
        if index == -1:
            index = 0
        self.ui.comboBox_text_chunk_size.setCurrentIndex(index)
        self.ui.checkBox_auto_backup.stateChanged.connect(self.backup_state_changed)
        if self.settings['showids'] == 'True':
            self.ui.checkBox.setChecked(True)
        else:
            self.ui.checkBox.setChecked(False)
        styles = ["original", "dark", "blue", "green", "orange", "purple", "yellow", "rainbow", "native"]
        styles_translated = [_("original"), _("dark"), _("blue"), _("green"), _("orange"), _("purple"), _("yellow"), _("rainbow"), _("native")]
        self.ui.comboBox_style.addItems(styles_translated)
        for index, style in enumerate(styles):
            if style == self.settings['stylesheet']:
                self.ui.comboBox_style.setCurrentIndex(index)
        if self.settings['backup_on_open'] == 'True':
            self.ui.checkBox_auto_backup.setChecked(True)
        else:
            self.ui.checkBox_auto_backup.setChecked(False)
        if self.settings['backup_av_files'] == 'True':
            self.ui.checkBox_backup_AV_files.setChecked(True)
        else:
            self.ui.checkBox_backup_AV_files.setChecked(False)
        self.ui.spinBox_backups.setValue(self.settings['backup_num'])
        if self.settings['directory'] == "":
            self.settings['directory'] = os.path.expanduser("~")
        self.ui.label_directory.setText(self.settings['directory'])
        text_styles = [_('Bold'), _('Italic'), _('Bigger')]
        self.ui.comboBox_text_style.addItems(text_styles)
        for index, text_style in enumerate(text_styles):
            if text_style == self.settings['report_text_context_style']:
                self.ui.comboBox_text_style.setCurrentIndex(index)
        self.ui.spinBox_chars_before_after.setValue(self.settings['report_text_context_characters'])
        self.ui.pushButton_choose_directory.clicked.connect(self.choose_directory)
        self.ui.pushButton_set_coder.pressed.connect(self.new_coder_entered)
        # AI options
        self.ui.checkBox_AI_enable.setChecked(self.settings['ai_enable'] == 'True')
        self.ui.checkBox_AI_enable.stateChanged.connect(self.ai_enable_state_changed)
        self.ui.comboBox_ai_model.clear()
        for i in range(len(self.ai_models)):
            model = self.ai_models[i]
            self.ui.comboBox_ai_model.addItem(model['name'])
            self.ui.comboBox_ai_model.setItemData(i, model['desc'], QtCore.Qt.ItemDataRole.ToolTipRole)
        curr_model = self.ai_models[int(self.settings['ai_model_index'])]
        self.ui.comboBox_ai_model.setCurrentIndex(int(self.settings['ai_model_index']))
        self.ui.comboBox_ai_model.currentTextChanged.connect(self.ai_model_changed)
        self.ai_model_changed()
        self.ai_enable_state_changed()
        self.ui.lineEdit_ai_api_key.textChanged.connect(self.ai_api_key_changed)
        self.ui.checkBox_ai_project_memo.setChecked(self.settings.get('ai_send_project_memo', 'True') == 'True')
        if section is not None and section == 'AI':
            self.ui.scrollArea.verticalScrollBar().setValue(self.ui.scrollArea.verticalScrollBar().maximum())
            # Use QTimers to briefly flash a yellow border around the AI settings
            QtCore.QTimer.singleShot(200, lambda:self.ui.widget_ai.setStyleSheet('#widget_ai {\n'
                                                                                  '   border: 3px solid yellow; \n'
                                                                                  '   border-radius: 5px; \n'
                                                                                  '}'))
            QtCore.QTimer.singleShot(700, lambda: self.ui.widget_ai.setStyleSheet('#widget_ai { border: none; }'))
        else:
            self.ui.widget_ai.setStyleSheet('')

    def backup_state_changed(self):
        """ Enable and disable av backup checkbox. Only enable when checkBox_auto_backup is checked. """

        if self.ui.checkBox_auto_backup.isChecked():
            self.ui.checkBox_backup_AV_files.setEnabled(True)
        else:
            self.ui.checkBox_backup_AV_files.setEnabled(False)
    
    def ai_enable_state_changed(self):
        self.ui.comboBox_ai_model.setEnabled(self.ui.checkBox_AI_enable.isChecked())
        self.ui.label_ai_model_desc.setEnabled(self.ui.checkBox_AI_enable.isChecked())
        self.ui.label_ai_access_info_url.setEnabled(self.ui.checkBox_AI_enable.isChecked())
        self.ui.lineEdit_ai_api_key.setEnabled(self.ui.checkBox_AI_enable.isChecked())
        self.ui.checkBox_ai_project_memo.setEnabled(self.ui.checkBox_AI_enable.isChecked())
    
    def ai_model_changed(self):
        self.settings['ai_model_index'] = self.ui.comboBox_ai_model.currentIndex()
        if self.settings['ai_model_index'] >= 0:
            curr_model = self.ai_models[int(self.settings['ai_model_index'])]
            self.ui.label_ai_model_desc.setText(curr_model['desc'])
            self.ui.label_ai_access_info_url.setText(f'<a href="{curr_model["access_info_url"]}">{curr_model["access_info_url"]}</a>')
            self.ui.lineEdit_ai_api_key.setText(curr_model['api_key'])
        else:
            curr_model = None
            self.ui.label_ai_model_desc.setText('')
            self.ui.label_ai_access_info_url.setText('')
            self.ui.lineEdit_ai_api_key.setText('')            

    def ai_api_key_changed(self):
        curr_model = self.ai_models[int(self.settings['ai_model_index'])]
        curr_model['api_key'] = self.ui.lineEdit_ai_api_key.text()        
                
    def new_coder_entered(self):
        """ New coder name entered.
        Tried to disable Enter key or catch the event. Failed. So new coder name assigned
        when the pushButton_set_coder is activated. """

        new_coder = self.ui.lineEdit_coderName.text()
        if new_coder == "":
            return
        self.ui.lineEdit_coderName.setEnabled(False)
        self.current_coder = new_coder
        self.ui.label_current_coder.setText(_("Current coder: ") + self.current_coder)

    def combobox_coder_changed(self):
        """ Set the coder name to the current selection. """

        current_selection = self.ui.comboBox_coders.currentText()
        if current_selection == "":
            return
        self.current_coder = current_selection
        self.ui.label_current_coder.setText(_("Current coder: ") + self.current_coder)

    def choose_directory(self):
        """ Choose default project directory. """

        directory = QtWidgets.QFileDialog.getExistingDirectory(self,
            _('Choose project directory'), self.settings['directory'])
        if directory == "":
            return
        self.ui.label_directory.setText(directory)

    def accept(self):
        restart_qualcoder = False
        self.settings['codername'] = self.current_coder
        if self.settings['codername'] == "":
            self.settings['codername'] = "default"
        if self.app.conn is not None:
            # None if no project opened
            cur = self.app.conn.cursor()
            cur.execute('update project set codername=?', [self.settings['codername']])
            self.app.conn.commit()
        self.settings['font'] = self.ui.fontComboBox.currentText()
        self.settings['fontsize'] = self.ui.spinBox.value()
        self.settings['treefontsize'] = self.ui.spinBox_treefontsize.value()
        self.settings['docfontsize'] = self.ui.spinBox_docfontsize.value()
        self.settings['directory'] = self.ui.label_directory.text()
        if self.ui.checkBox.isChecked():
            self.settings['showids'] = 'True'
        else:
            self.settings['showids'] = 'False'
        index = self.ui.comboBox_style.currentIndex()
        styles = ["original", "dark", "blue", "green", "orange", "purple", "yellow", "rainbow", "native"]
        if self.settings['stylesheet'] != styles[index]:
            restart_qualcoder = True
        self.settings['stylesheet'] = styles[index]
        if self.settings['language'] != self.ui.comboBox_language.currentText()[-2:]:
            restart_qualcoder = True
        self.settings['language'] = self.ui.comboBox_language.currentText()[-2:]
        self.settings['codetext_chunksize'] = int(self.ui.comboBox_text_chunk_size.currentText())
        self.settings['timestampformat'] = self.ui.comboBox_timestamp.currentText()
        self.settings['speakernameformat'] = self.ui.comboBox_speaker.currentText()
        if self.ui.checkBox_auto_backup.isChecked():
            self.settings['backup_on_open'] = 'True'
        else:
            self.settings['backup_on_open'] = 'False'
        if self.ui.checkBox_backup_AV_files.isChecked():
            self.settings['backup_av_files'] = 'True'
        else:
            self.settings['backup_av_files'] = 'False'
        self.settings['backup_num'] = self.ui.spinBox_backups.value()
        self.settings['report_text_context_characters'] = self.ui.spinBox_chars_before_after.value()
        ts_index = self.ui.comboBox_text_style.currentIndex()
        self.settings['report_text_context_style'] = ['Bold', 'Italic', 'Bigger'][ts_index]
        # AI settings
        if self.ui.checkBox_AI_enable.isChecked():
            self.settings['ai_enable'] = 'True'
        else:
            self.settings['ai_enable'] = 'False'
        ai_model_index = self.ui.comboBox_ai_model.currentIndex() 
        self.settings['ai_model_index'] = ai_model_index
        if self.settings['ai_enable'] == 'True' and ai_model_index < 0:
            msg = _('Please select an AI model or disable the AI altogether.')
            Message(self.app, _('AI model'), msg).exec()
            return
        if self.settings['ai_enable'] == 'True' and self.ai_models[ai_model_index]['api_key'] == '':
            msg = _('Please enter a valid API-key for the AI model. \n(If you are sure that your particular model does not need an API-key, enter "None" instead.)')
            Message(self.app, _('AI model'), msg).exec()
            return
        if self.ui.checkBox_ai_project_memo.isChecked():
            self.settings['ai_send_project_memo'] = 'True'
        else: 
            self.settings['ai_send_project_memo'] = 'False'
        self.save_settings()
        if restart_qualcoder:
            Message(self.app, _("Restart QualCoder"), _("Restart QualCoder to enact some changes")).exec()
        super().accept()

    def save_settings(self):
        """ Save settings to text file in user's home directory.
        Each setting has a variable identifier then a colon
        followed by the value. """

        self.app.write_config_ini(self.settings, self.ai_models)
