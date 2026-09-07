'use client';
import { useEffect, useRef, useState } from 'react';
import { Compartment, EditorState } from '@codemirror/state';
import { EditorView, keymap } from '@codemirror/view';
import { basicSetup } from 'codemirror';
import { cpp } from '@codemirror/lang-cpp';
import { oneDark } from '@codemirror/theme-one-dark';
import { indentWithTab } from '@codemirror/commands';

export function CodeEditor({
  value,
  onChange,
  onRun,
  readOnly = false,
}: {
  value: string;
  readOnly?: boolean;
  onChange: (value: string) => void;
  onRun: () => void;
}) {
  const [editable] = useState(() => new Compartment());
  const parent = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const change = useRef(onChange);
  const run = useRef(onRun);
  useEffect(() => {
    change.current = onChange;
    run.current = onRun;
  }, [onChange, onRun]);
  useEffect(() => {
    if (!parent.current) return;
    const editor = new EditorView({
      parent: parent.current,
      state: EditorState.create({
        doc: '',
        extensions: [
          basicSetup,
          editable.of([
            EditorView.editable.of(true),
            EditorState.readOnly.of(false),
          ]),
          cpp(),
          oneDark,
          keymap.of([
            {
              key: 'Mod-Enter',
              run: () => {
                run.current();
                return true;
              },
            },
            indentWithTab,
          ]),
          EditorView.contentAttributes.of({
            'aria-label': 'C++ code editor',
            spellcheck: 'false',
          }),
          EditorState.tabSize.of(4),
          EditorView.theme({
            '&': {
              backgroundColor: '#141e31',
              fontSize: '0.875rem',
              height: '100%',
            },
            '.cm-scroller': {
              fontFamily: 'var(--font-mono)',
              lineHeight: '1.8',
              overflow: 'auto',
            },
            '.cm-content': { padding: '17px 0', minHeight: '330px' },
            '.cm-gutters': {
              backgroundColor: '#141e31',
              color: '#667b99',
              border: 'none',
              paddingRight: '8px',
            },
            '.cm-activeLine': { backgroundColor: '#202d4380' },
            '.cm-activeLineGutter': { backgroundColor: '#202d4380' },
            '&.cm-focused': { outline: 'none' },
          }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) change.current(update.state.doc.toString());
          }),
        ],
      }),
    });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
  }, [editable]);
  useEffect(() => {
    view.current?.dispatch({
      effects: editable.reconfigure([
        EditorView.editable.of(!readOnly),
        EditorState.readOnly.of(readOnly),
      ]),
    });
  }, [readOnly, editable]);
  useEffect(() => {
    const editor = view.current;
    if (editor && editor.state.doc.toString() !== value)
      editor.dispatch({
        changes: { from: 0, to: editor.state.doc.length, insert: value },
      });
  }, [value]);
  return <div ref={parent} className="codemirror-container" />;
}
