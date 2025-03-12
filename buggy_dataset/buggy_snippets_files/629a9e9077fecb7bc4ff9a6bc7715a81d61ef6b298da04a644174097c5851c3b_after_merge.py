    def setup_page(self):
        template_btn = self.create_button(_("Edit template for new modules"),
                                          self.plugin.edit_template)

        interface_group = QGroupBox(_("Interface"))
        newcb = self.create_checkbox
        showtabbar_box = newcb(_("Show tab bar"), 'show_tab_bar')
        showclassfuncdropdown_box = newcb(
                _("Show selector for classes and functions"),
                'show_class_func_dropdown')
        showindentguides_box = newcb(_("Show Indent Guides"),
                                     'indent_guides')

        interface_layout = QVBoxLayout()
        interface_layout.addWidget(showtabbar_box)
        interface_layout.addWidget(showclassfuncdropdown_box)
        interface_layout.addWidget(showindentguides_box)
        interface_group.setLayout(interface_layout)

        display_group = QGroupBox(_("Source code"))
        linenumbers_box = newcb(_("Show line numbers"), 'line_numbers')
        blanks_box = newcb(_("Show blank spaces"), 'blank_spaces')
        edgeline_box = newcb(_("Show vertical lines after"), 'edge_line')
        edgeline_edit = self.create_lineedit("", 'edge_line_columns',
                                             tip=("Enter values separated by "
                                                  "commas ','"),
                                             alignment=Qt.Horizontal,
                                             regex="[0-9]+(,[0-9]+)*")
        edgeline_edit_label = QLabel(_("characters"))
        edgeline_box.toggled.connect(edgeline_edit.setEnabled)
        edgeline_box.toggled.connect(edgeline_edit_label.setEnabled)
        edgeline_edit.setEnabled(self.get_option('edge_line'))
        edgeline_edit_label.setEnabled(self.get_option('edge_line'))

        currentline_box = newcb(_("Highlight current line"),
                                'highlight_current_line')
        currentcell_box = newcb(_("Highlight current cell"),
                                'highlight_current_cell')
        occurrence_box = newcb(_("Highlight occurrences after"),
                               'occurrence_highlighting')
        occurrence_spin = self.create_spinbox(
            "", _(" ms"),
            'occurrence_highlighting/timeout',
            min_=100, max_=1000000, step=100)
        occurrence_box.toggled.connect(occurrence_spin.spinbox.setEnabled)
        occurrence_box.toggled.connect(occurrence_spin.slabel.setEnabled)
        occurrence_spin.spinbox.setEnabled(
                self.get_option('occurrence_highlighting'))
        occurrence_spin.slabel.setEnabled(
                self.get_option('occurrence_highlighting'))

        wrap_mode_box = newcb(_("Wrap lines"), 'wrap')
        scroll_past_end_box = newcb(_("Scroll past the end"),
                                    'scroll_past_end')

        display_layout = QGridLayout()
        display_layout.addWidget(linenumbers_box, 0, 0)
        display_layout.addWidget(blanks_box, 1, 0)
        display_layout.addWidget(edgeline_box, 2, 0)
        display_layout.addWidget(edgeline_edit, 2, 1)
        display_layout.addWidget(edgeline_edit_label, 2, 2)
        display_layout.addWidget(currentline_box, 3, 0)
        display_layout.addWidget(currentcell_box, 4, 0)
        display_layout.addWidget(occurrence_box, 5, 0)
        display_layout.addWidget(occurrence_spin.spinbox, 5, 1)
        display_layout.addWidget(occurrence_spin.slabel, 5, 2)
        display_layout.addWidget(wrap_mode_box, 6, 0)
        display_layout.addWidget(scroll_past_end_box, 7, 0)
        display_h_layout = QHBoxLayout()
        display_h_layout.addLayout(display_layout)
        display_h_layout.addStretch(1)
        display_group.setLayout(display_h_layout)

        run_group = QGroupBox(_("Run"))
        saveall_box = newcb(_("Save all files before running script"),
                            'save_all_before_run')

        run_selection_group = QGroupBox(_("Run selection"))
        focus_box = newcb(_("Maintain focus in the Editor after running cells "
                            "or selections"), 'focus_to_editor')
        run_cell_box = newcb(_("Copy full cell contents to the console when "
                               "running code cells"), 'run_cell_copy')

        sourcecode_group = QGroupBox(_("Source code"))
        closepar_box = newcb(_("Automatic insertion of parentheses, braces "
                               "and brackets"),
                             'close_parentheses')
        close_quotes_box = newcb(_("Automatic insertion of closing quotes"),
                                 'close_quotes')
        add_colons_box = newcb(_("Automatic insertion of colons after 'for', "
                                 "'if', 'def', etc"),
                               'add_colons')
        autounindent_box = newcb(_("Automatic indentation after 'else', "
                                   "'elif', etc."), 'auto_unindent')
        indent_chars_box = self.create_combobox(_("Indentation characters: "),
                                                ((_("2 spaces"), '*  *'),
                                                 (_("3 spaces"), '*   *'),
                                                 (_("4 spaces"), '*    *'),
                                                 (_("5 spaces"), '*     *'),
                                                 (_("6 spaces"), '*      *'),
                                                 (_("7 spaces"), '*       *'),
                                                 (_("8 spaces"), '*        *'),
                                                 (_("Tabulations"), '*\t*')),
                                                'indent_chars')
        tabwidth_spin = self.create_spinbox(_("Tab stop width:"), _("spaces"),
                                            'tab_stop_width_spaces',
                                            4, 1, 8, 1)

        def enable_tabwidth_spin(index):
            if index == 7:  # Tabulations
                tabwidth_spin.plabel.setEnabled(True)
                tabwidth_spin.spinbox.setEnabled(True)
            else:
                tabwidth_spin.plabel.setEnabled(False)
                tabwidth_spin.spinbox.setEnabled(False)

        indent_chars_box.combobox.currentIndexChanged.connect(
            enable_tabwidth_spin)

        tab_mode_box = newcb(
            _("Tab always indent"),
            'tab_always_indent', default=False,
            tip=_("If enabled, pressing Tab will always indent,\n"
                  "even when the cursor is not at the beginning\n"
                  "of a line (when this option is enabled, code\n"
                  "completion may be triggered using the alternate\n"
                  "shortcut: Ctrl+Space)"))
        ibackspace_box = newcb(_("Intelligent backspace"),
                               'intelligent_backspace', default=True)
        removetrail_box = newcb(_("Automatically remove trailing spaces "
                                  "when saving files"),
                                'always_remove_trailing_spaces', default=False)
        docstring_combo_choices = ((_("Numpydoc"), 'Numpydoc'),
                                   (_("Googledoc"), 'Googledoc'),)
        docstring_combo = self.create_combobox("Docstring type",
                                               docstring_combo_choices,
                                               'docstring_type')

        analysis_group = QGroupBox(_("Analysis"))
        todolist_box = newcb(_("Code annotations (TODO, FIXME, XXX, HINT, TIP,"
                               " @todo, HACK, BUG, OPTIMIZE, !!!, ???)"),
                             'todo_list', default=True)

        run_layout = QVBoxLayout()
        run_layout.addWidget(saveall_box)
        run_group.setLayout(run_layout)

        run_selection_layout = QVBoxLayout()
        run_selection_layout.addWidget(focus_box)
        run_selection_layout.addWidget(run_cell_box)
        run_selection_group.setLayout(run_selection_layout)

        analysis_layout = QVBoxLayout()
        analysis_layout.addWidget(todolist_box)
        analysis_group.setLayout(analysis_layout)

        sourcecode_layout = QVBoxLayout()
        sourcecode_layout.addWidget(closepar_box)
        sourcecode_layout.addWidget(autounindent_box)
        sourcecode_layout.addWidget(add_colons_box)
        sourcecode_layout.addWidget(close_quotes_box)
        indent_tab_layout = QHBoxLayout()
        indent_tab_grid_layout = QGridLayout()
        indent_tab_grid_layout.addWidget(indent_chars_box.label, 0, 0)
        indent_tab_grid_layout.addWidget(indent_chars_box.combobox, 0, 1)
        indent_tab_grid_layout.addWidget(tabwidth_spin.plabel, 1, 0)
        indent_tab_grid_layout.addWidget(tabwidth_spin.spinbox, 1, 1)
        indent_tab_grid_layout.addWidget(tabwidth_spin.slabel, 1, 2)
        indent_tab_layout.addLayout(indent_tab_grid_layout)
        indent_tab_layout.addStretch(1)
        sourcecode_layout.addLayout(indent_tab_layout)
        sourcecode_layout.addWidget(tab_mode_box)
        sourcecode_layout.addWidget(ibackspace_box)
        sourcecode_layout.addWidget(removetrail_box)
        sourcecode_layout.addWidget(docstring_combo)
        sourcecode_group.setLayout(sourcecode_layout)

        eol_group = QGroupBox(_("End-of-line characters"))
        eol_label = QLabel(_("When opening a text file containing "
                             "mixed end-of-line characters (this may "
                             "raise syntax errors in the consoles "
                             "on Windows platforms), Spyder may fix the "
                             "file automatically."))
        eol_label.setWordWrap(True)
        check_eol_box = newcb(_("Fix automatically and show warning "
                                "message box"),
                              'check_eol_chars', default=True)
        convert_eol_on_save_box = newcb(_("On Save: convert EOL characters"
                                          " to"),
                                        'convert_eol_on_save', default=False)
        eol_combo_choices = ((_("LF (UNIX)"), 'LF'),
                             (_("CRLF (Windows)"), 'CRLF'),
                             (_("CR (Mac)"), 'CR'),
                             )
        convert_eol_on_save_combo = self.create_combobox("",
                                                         eol_combo_choices,
                                                         ('convert_eol_on_'
                                                          'save_to'),
                                                         )
        convert_eol_on_save_box.toggled.connect(
                convert_eol_on_save_combo.setEnabled)
        convert_eol_on_save_combo.setEnabled(
                self.get_option('convert_eol_on_save'))

        eol_on_save_layout = QHBoxLayout()
        eol_on_save_layout.addWidget(convert_eol_on_save_box)
        eol_on_save_layout.addWidget(convert_eol_on_save_combo)

        eol_layout = QVBoxLayout()
        eol_layout.addWidget(eol_label)
        eol_layout.addWidget(check_eol_box)
        eol_layout.addLayout(eol_on_save_layout)
        eol_group.setLayout(eol_layout)

        autosave_group = QGroupBox(_('Autosave'))
        autosave_checkbox = newcb(
                _('Automatically save a copy of files with unsaved changes'),
                'autosave_enabled', default=True)
        autosave_spinbox = self.create_spinbox(
                _('Autosave interval: '), _('seconds'), 'autosave_interval',
                min_=1, max_=3600)
        autosave_checkbox.toggled.connect(autosave_spinbox.setEnabled)

        autosave_layout = QVBoxLayout()
        autosave_layout.addWidget(autosave_checkbox)
        autosave_layout.addWidget(autosave_spinbox)
        autosave_group.setLayout(autosave_layout)

        tabs = QTabWidget()
        tabs.addTab(self.create_tab(interface_group, display_group),
                    _("Display"))
        tabs.addTab(self.create_tab(template_btn, run_group,
                                    run_selection_group, sourcecode_group,
                                    eol_group, autosave_group, analysis_group),
                    _("Advanced settings"))

        vlayout = QVBoxLayout()
        vlayout.addWidget(tabs)
        self.setLayout(vlayout)