'use client';
import { useEffect, useRef } from 'react';
import {
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronRight,
  CircleCheck,
  CircleX,
  Clock3,
  Coffee,
  ExternalLink,
  LoaderCircle,
  LockKeyhole,
  Send,
  Sparkles,
  Terminal,
  Zap,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { Challenge, Message, RunResult } from '@/lib/quest-types';
import { trackIcons, trackNames } from './learning-views';
export function ProblemPanel({
  challenge,
  onBrowse,
}: {
  challenge: Challenge;
  onBrowse: () => void;
}) {
  const c = challenge;
  const Icon = c.track === 'foundations' ? Coffee : trackIcons[c.track];
  return (
    <section className="problem-panel">
      <div className="mission-banner">
        <span className={'mission-icon track-' + c.track}>
          <Icon size={27} />
        </span>
        <div>
          <div className="eyebrow">
            MISSION {String(c.order).padStart(2, '0')} · {c.track}
          </div>
          <h2>{c.title}</h2>
        </div>
      </div>
      <div className="mission-meta">
        <span className={'difficulty ' + c.difficulty}>{c.difficulty}</span>
        <span>
          <Clock3 size={14} />
          {c.minutes} min
        </span>
        <span className="xp-text">
          <Zap size={14} />
          {c.xp} XP
        </span>
      </div>
      <Tabs defaultValue="challenge" key={c.id}>
        <TabsList variant="line" className="problem-tabs">
          <TabsTrigger value="challenge">Challenge</TabsTrigger>
          <TabsTrigger value="concepts">Learn the concept</TabsTrigger>
        </TabsList>
        <TabsContent value="challenge">
          <div className="story-tag">
            <Icon size={14} />
            REAL-WORLD SCENARIO
          </div>
          <p>{c.story}</p>
          <h3>Your mission</h3>
          <p className="preserve-lines">{c.prompt}</p>
          <h3>Input</h3>
          <p>{c.input_format}</p>
          <h3>Output</h3>
          <p>{c.output_format}</p>
          <h3>Examples</h3>
          {c.tests.map((t, i) => (
            <div className="example-box" key={i}>
              <div>
                <small>INPUT {i + 1}</small>
                <pre>{t.input.trimEnd() || '(empty)'}</pre>
              </div>
              <ArrowUpRight size={16} />
              <div>
                <small>OUTPUT</small>
                <pre>{t.output.trimEnd() || '(empty)'}</pre>
              </div>
            </div>
          ))}
          <h3>Keep in mind</h3>
          <ul className="constraint-list">
            {c.constraints.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
          <h3>What you’ll practice</h3>
          <div className="objectives">
            {c.objectives.map((s, i) => (
              <span key={i}>
                <Check size={16} />
                {s}
              </span>
            ))}
          </div>
        </TabsContent>
        <TabsContent value="concepts">
          {c.concepts.map((concept, i) => (
            <article className="concept" key={i}>
              <h3>{concept.title}</h3>
              <p>{concept.body}</p>
              {concept.code && (
                <pre className="concept-code">{concept.code}</pre>
              )}
            </article>
          ))}
        </TabsContent>
      </Tabs>
      <button className="problem-bottom" onClick={onBrowse}>
        <span>
          <BookOpen size={16} />
          {trackNames[c.track]}
        </span>
        <ChevronRight size={16} />
      </button>
    </section>
  );
}
const statusLabels: Record<string, string> = {
  passed: 'Passed',
  wrong_answer: 'Output differs',
  time_limit: 'Time limit',
  output_limit: 'Output limit',
  memory_limit: 'Memory limit',
  runtime_error: 'Runtime error',
  executed: 'Finished',
};
export function ConsolePanel({
  result,
  running,
  onDebug,
  stale,
}: {
  result: RunResult | null;
  stale: boolean;
  running: boolean;
  onDebug: () => void;
}) {
  return (
    <section
      className="console-panel"
      aria-label="Code results"
      aria-live="polite"
    >
      <div className="console-header">
        <span>
          <Terminal size={16} />
          Results
        </span>
        <span className="console-count">
          {running
            ? 'COMPILING & RUNNING'
            : result
              ? result.mode === 'submit'
                ? 'ALL TESTS'
                : result.mode === 'custom'
                  ? 'CUSTOM INPUT'
                  : 'SAMPLE TESTS'
              : 'READY'}
        </span>
      </div>
      {running ? (
        <div className="console-empty">
          <LoaderCircle className="spin" size={26} />
          <h3>Your program is running locally.</h3>
          <p>First launch can take a few seconds.</p>
        </div>
      ) : !result ? (
        <div className="console-empty">
          <Terminal size={27} />
          <h3>Your next “it works!” starts here.</h3>
          <p>Run the examples, then submit against every test.</p>
        </div>
      ) : (
        <div className="console-results">
          {stale && (
            <p className="stale-result">
              Your code changed after this run. These results describe your
              earlier draft.
            </p>
          )}
          {result.compile_status !== 'ok' ? (
            <div className="compile-error">
              <strong>
                <CircleX size={18} />
                Compilation did not finish
              </strong>
              <pre>
                {result.diagnostics ||
                  'The compiler could not run. Check the local engine status.'}
              </pre>
            </div>
          ) : (
            <>
              <div
                className={
                  'result-summary ' +
                  (result.passed === result.total ? 'success' : 'failure')
                }
              >
                <strong>
                  {result.mode === 'custom'
                    ? result.passed
                      ? 'Program finished'
                      : 'Program stopped'
                    : `${result.passed} of ${result.total} tests passed`}
                </strong>
                <span>{(result.duration_ms / 1000).toFixed(1)} s total</span>
              </div>
              {result.results.map((row) => (
                <details
                  className={
                    'case-row ' + (row.passed ? 'case-pass' : 'case-fail')
                  }
                  key={row.index}
                  open={!row.passed && !row.hidden}
                >
                  <summary>
                    {row.passed ? (
                      <CircleCheck size={17} />
                    ) : (
                      <CircleX size={17} />
                    )}
                    <b>
                      {row.hidden ? 'Hidden test' : 'Test'} {row.index}
                    </b>
                    {row.hidden && <LockKeyhole size={13} />}
                    <span>
                      {statusLabels[row.status] ||
                        row.status.replaceAll('_', ' ')}
                    </span>
                    {typeof row.time_ms === 'number' && (
                      <small>{row.time_ms.toFixed(1)} ms</small>
                    )}
                  </summary>
                  {row.hidden ? (
                    <p className="hidden-note">
                      Private case. Use the constraints to reason about edge
                      cases.
                    </p>
                  ) : (
                    <div className="case-detail">
                      <div>
                        <small>INPUT</small>
                        <pre>{row.input || '(empty)'}</pre>
                      </div>
                      {row.expected !== undefined && (
                        <div>
                          <small>EXPECTED</small>
                          <pre>{row.expected || '(empty)'}</pre>
                        </div>
                      )}
                      <div>
                        <small>YOUR OUTPUT</small>
                        <pre>{row.actual || row.stdout || '(empty)'}</pre>
                      </div>
                      {row.stderr && (
                        <div>
                          <small>DIAGNOSTIC</small>
                          <pre>{row.stderr}</pre>
                        </div>
                      )}
                    </div>
                  )}
                </details>
              ))}
              {result.diagnostics && (
                <details className="compiler-notes">
                  <summary>Compiler notes</summary>
                  <pre>{result.diagnostics}</pre>
                </details>
              )}
            </>
          )}
          {(result.compile_status !== 'ok' || result.passed < result.total) && (
            <Button variant="ghost" className="debug-button" onClick={onDebug}>
              <Sparkles size={15} />
              Help me understand this result
              <ChevronRight size={15} />
            </Button>
          )}
          {result.mode === 'run' && result.passed === result.total && (
            <p className="submit-reminder">
              Examples passed. Submit to check hidden cases and earn XP.
            </p>
          )}
          {result.mode === 'submit' && result.passed === result.total && (
            <p className="submit-reminder">
              {result.xp_awarded
                ? `Mission complete · +${result.xp_awarded} XP`
                : 'Mission complete. XP is awarded once per mission.'}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
function SafeText({ text }: { text: string }) {
  return (
    <>
      {text.split(/```(?:[a-zA-Z0-9_+-]*)\n?/g).map((part, i) =>
        i % 2 ? (
          <pre className="chat-code" key={i}>
            {part.trimEnd()}
          </pre>
        ) : (
          <div className="chat-prose" key={i}>
            {part
              .split(/\n\n+/)
              .filter(Boolean)
              .map((p, j) => (
                <p key={j}>{p}</p>
              ))}
          </div>
        ),
      )}
    </>
  );
}
export function MentorPanel({
  challenge,
  name,
  messages,
  question,
  setQuestion,
  onAsk,
  busy,
  mode,
  onMode,
  experimentalAvailable,
}: {
  challenge: Challenge;
  name: string;
  messages: Message[];
  question: string;
  setQuestion: (s: string) => void;
  onAsk: (s: string) => void;
  busy: boolean;
  mode: 'grounded' | 'experimental';
  onMode: (s: 'grounded' | 'experimental') => void;
  experimentalAvailable: boolean;
}) {
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = bottom.current?.parentElement;
    if (container) container.scrollTop = container.scrollHeight;
  }, [messages, busy]);
  const suggestions = messages.filter((m) => m.role === 'assistant').at(-1)
    ?.suggestions || [
    'Give me a small hint',
    'Explain this concept',
    'Review my approach',
  ];
  return (
    <aside className="mentor-panel">
      <div className="mentor-header">
        <span className="mentor-orb">
          <Sparkles size={21} />
        </span>
        <div>
          <h2>Your study mentor</h2>
          <span>
            <i />
            Here to help you think
          </span>
        </div>
      </div>
      <div className="mentor-mode">
        <Select
          value={mode}
          onValueChange={(v) => onMode(v as 'grounded' | 'experimental')}
          disabled={busy}
        >
          <SelectTrigger aria-label="Mentor mode">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="grounded">Grounded study guide</SelectItem>
            <SelectItem value="experimental" disabled={!experimentalAvailable}>
              Your NumPy model · experimental
            </SelectItem>
          </SelectContent>
        </Select>
      </div>
      {mode === 'experimental' && (
        <p className="experimental-notice">
          Your tiny model runs locally. It has not passed the C++ knowledge
          test; treat its output as a training experiment.
        </p>
      )}
      <div className="mentor-chat" role="log" aria-label="Mentor conversation">
        <div className="mentor-note">
          <p>Hey, {name} 👋</p>
          <p>
            Let’s work through <strong>{challenge.title.toLowerCase()}</strong>.
            Read the example, try an approach, and I’ll help you reason through
            it.
          </p>
          <p>Ask for a hint when you need a nudge.</p>
        </div>
        {messages.map((m, i) => (
          <div
            key={m.id !== undefined ? 'stored-' + m.id : 'local-' + i}
            className={'chat-message ' + m.role}
          >
            <span className="chat-author">
              {m.role === 'user'
                ? 'You'
                : m.model?.mode === 'experimental'
                  ? 'Experimental NumPy model'
                  : 'Study mentor'}
            </span>
            <SafeText text={m.text} />
            {m.sources && m.sources.length > 0 && (
              <div className="chat-sources">
                {m.sources
                  .filter((s) => s.url.startsWith('https://'))
                  .map((s, j) => (
                    <a
                      key={j}
                      href={s.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {s.title}
                      <ExternalLink size={12} />
                    </a>
                  ))}
              </div>
            )}
          </div>
        ))}
        {busy && (
          <div className="mentor-thinking" aria-live="polite">
            <LoaderCircle size={16} className="spin" />
            Working through your question…
          </div>
        )}
        <div ref={bottom} />
      </div>
      <div className="mentor-chips">
        {suggestions.slice(0, 3).map((s) => (
          <button
            key={s}
            disabled={busy}
            onClick={() => {
              if (mode === 'experimental' && s.startsWith('Switch to guided'))
                onMode('grounded');
              else onAsk(s);
            }}
          >
            {s}
            <ChevronRight size={14} />
          </button>
        ))}
      </div>
      <form
        className="mentor-compose"
        onSubmit={(e) => {
          e.preventDefault();
          onAsk(question);
        }}
      >
        <textarea
          placeholder="Ask about this mission…"
          aria-label="Ask your mentor"
          maxLength={1500}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (!busy && question.trim()) onAsk(question);
            }
          }}
        />
        <button
          aria-label="Send message"
          disabled={busy || !question.trim()}
          type="submit"
        >
          <Send size={17} />
        </button>
      </form>
      <small className="mentor-footnote">
        Local guidance · Code stays on your Mac
      </small>
    </aside>
  );
}
