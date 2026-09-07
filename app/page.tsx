'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronRight,
  Circle,
  CodeXml,
  Flame,
  Layers,
  Leaf,
  LoaderCircle,
  Map,
  Play,
  RotateCcw,
  Search,
  Settings2,
  ShieldCheck,
  Trophy,
  X,
  Zap,
} from 'lucide-react';
import {
  Sidebar,
  SidebarProvider,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarTrigger,
} from '@/components/ui/sidebar';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import {
  Achievements,
  Catalog,
  Dashboard,
  trackIcons,
  trackNames,
} from '@/components/quest/learning-views';
import { CodeEditor } from '@/components/quest/code-editor';
import {
  ConsolePanel,
  MentorPanel,
  ProblemPanel,
} from '@/components/quest/practice-panels';
import { api, bootstrap } from '@/lib/quest-api';
import type {
  Bootstrap,
  Detail,
  Message,
  Profile,
  RunResult,
  TrackId,
  ViewName,
} from '@/lib/quest-types';
const nav = [
  { id: 'dashboard', label: 'Dashboard', icon: Map },
  { id: 'practice', label: 'Practice arena', icon: CodeXml },
  { id: 'labs', label: 'Real-world labs', icon: Layers },
  { id: 'achievements', label: 'Achievements', icon: Trophy },
] as const;
const errorText = (e: unknown) =>
  e instanceof Error
    ? e.message
    : 'Something interrupted the request. Try again.';
export default function Home() {
  const [data, setData] = useState<Bootstrap | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [view, setView] = useState<ViewName>('practice');
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [notice, setNotice] = useState('');
  const [saveStatus, setSaveStatus] = useState('Saved locally');
  const [saveRevision, setSaveRevision] = useState(0);
  const [running, setRunning] = useState(false);
  const [mentorBusy, setMentorBusy] = useState(false);
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [result, setResult] = useState<RunResult | null>(null);
  const [resultStale, setResultStale] = useState(false);
  const [inputMode, setInputMode] = useState('samples');
  const [stdin, setStdin] = useState('');
  const [filter, setFilter] = useState('all');
  const [track, setTrack] = useState('all');
  const [search, setSearch] = useState('');
  const [picker, setPicker] = useState(false);
  const [pickerSearch, setPickerSearch] = useState('');
  const [settings, setSettings] = useState(false);
  const [award, setAward] = useState(0);
  const [mode, setMode] = useState<'grounded' | 'experimental'>('grounded');
  const dataRef = useRef(data);
  const detailRef = useRef(detail);
  const codeRef = useRef(code);
  const busyRef = useRef(false);
  const switchingRef = useRef(false);
  const pendingSaves = useRef(0);
  const mentorRef = useRef(false);
  const saved = useRef({ id: '', source: '' });
  const saves = useRef<Promise<unknown>>(Promise.resolve());
  const loadSequence = useRef(0);
  const settingsWrites = useRef<Promise<unknown>>(Promise.resolve());
  const modeRevision = useRef(0);
  const persistedMode = useRef<'grounded' | 'experimental'>('grounded');
  useEffect(() => {
    dataRef.current = data;
    detailRef.current = detail;
    codeRef.current = code;
  }, [data, detail, code]);
  const save = useCallback((id: string, source: string) => {
    pendingSaves.current += 1;
    const task = saves.current
      .catch(() => {})
      .then(() => api('/api/draft', { challenge_id: id, source }))
      .then((response) => {
        if (detailRef.current?.challenge.id === id) {
          saved.current = { id, source };
          setSaveStatus(
            codeRef.current === source ? 'Saved locally' : 'Unsaved changes',
          );
          setSaveRevision((revision) => revision + 1);
        }
        return response;
      })
      .finally(() => {
        pendingSaves.current -= 1;
      });
    saves.current = task;
    return task;
  }, []);
  const loadMission = useCallback(
    async (id: string, initial = false) => {
      if (busyRef.current || mentorRef.current || switchingRef.current) {
        setNotice(
          'Finish the current run or mentor reply before opening another mission.',
        );
        return null;
      }
      const sequence = ++loadSequence.current;
      switchingRef.current = true;
      setSwitching(true);
      setNotice('');
      try {
        const previous = detailRef.current;
        if (previous && !initial) {
          await save(previous.challenge.id, codeRef.current);
        }
        const next = await api<Detail>(
          '/api/challenges/' + encodeURIComponent(id),
        );
        if (sequence !== loadSequence.current) return null;
        saved.current = { id, source: next.source };
        setDetail(next);
        detailRef.current = next;
        codeRef.current = next.source;
        setCode(next.source);
        setResult(next.last_run);
        setMessages(next.messages);
        setQuestion('');
        setInputMode('samples');
        setStdin(next.challenge.tests[0]?.input || '');
        setSaveStatus('Saved locally');
        setView('practice');
        setPicker(false);
        await save(id, next.source);
        return next;
      } catch (e) {
        setNotice(errorText(e));
        return null;
      } finally {
        if (sequence === loadSequence.current) {
          switchingRef.current = false;
          setSwitching(false);
        }
      }
    },
    [save],
  );
  const connect = useCallback(async () => {
    setLoading(true);
    setNotice('');
    try {
      const b = await bootstrap();
      setData(b);
      dataRef.current = b;
      setMode(b.profile.mentor_mode);
      persistedMode.current = b.profile.mentor_mode;
      await loadMission(b.profile.last_challenge || b.challenges[0].id, true);
    } catch (e) {
      setNotice(errorText(e));
    } finally {
      setLoading(false);
    }
  }, [loadMission]);
  useEffect(() => {
    let active = true;
    void Promise.resolve().then(() => {
      if (active) void connect();
    });
    return () => {
      active = false;
    };
  }, [connect]);
  useEffect(() => {
    if (
      !detail ||
      switching ||
      (saved.current.id === detail.challenge.id &&
        saved.current.source === code)
    )
      return;
    const id = detail.challenge.id;
    const timer = setTimeout(() => {
      setSaveStatus('Saving…');
      void save(id, code).catch(() =>
        setSaveStatus('Save interrupted · retry by running'),
      );
    }, 650);
    return () => clearTimeout(timer);
  }, [code, detail, switching, save, saveRevision]);
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (
        detailRef.current &&
        (pendingSaves.current > 0 || saved.current.source !== codeRef.current)
      ) {
        event.preventDefault();
      }
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, []);
  useEffect(() => {
    let active = true;
    const bytes = new TextEncoder().encode(code);
    void crypto.subtle
      .digest('SHA-256', bytes)
      .then((buffer) => {
        const hash = Array.from(new Uint8Array(buffer), (byte) =>
          byte.toString(16).padStart(2, '0'),
        ).join('');
        if (active)
          setResultStale(Boolean(result && result.source_sha256 !== hash));
      })
      .catch(() => {
        if (active) setResultStale(Boolean(result));
      });
    return () => {
      active = false;
    };
  }, [code, result]);
  const run = useCallback(
    async (runMode: 'run' | 'submit' | 'custom') => {
      const current = detailRef.current;
      if (!current || busyRef.current || switchingRef.current) return null;
      busyRef.current = true;
      setRunning(true);
      setNotice('');
      try {
        await saves.current.catch(() => {});
        const source = codeRef.current;
        const r = await api<RunResult>('/api/run', {
          challenge_id: current.challenge.id,
          source,
          mode: runMode,
          stdin,
        });
        if (detailRef.current?.challenge.id !== current.challenge.id) return r;
        setResult(r);
        if (r.profile)
          setData((b) =>
            b
              ? {
                  ...b,
                  profile: {
                    ...r.profile!,
                    name: b.profile.name,
                    difficulty: b.profile.difficulty,
                    track: b.profile.track,
                    daily_goal: b.profile.daily_goal,
                    mentor_mode: b.profile.mentor_mode,
                  },
                }
              : b,
          );
        await save(current.challenge.id, source);
        if (r.xp_awarded) setAward(r.xp_awarded);
        return r;
      } catch (e) {
        setNotice(errorText(e));
        return null;
      } finally {
        busyRef.current = false;
        setRunning(false);
      }
    },
    [stdin, save],
  );
  const ask = useCallback(
    async (text: string) => {
      const current = detailRef.current;
      if (!current || !text.trim() || mentorRef.current || switchingRef.current)
        return null;
      mentorRef.current = true;
      setMentorBusy(true);
      setQuestion('');
      setNotice('');
      try {
        const response = await api<Message>('/api/mentor', {
          challenge_id: current.challenge.id,
          source: codeRef.current,
          question: text.trim(),
          mode,
        });
        if (detailRef.current?.challenge.id !== current.challenge.id)
          return response;
        setMessages((previous) =>
          [
            ...previous,
            { role: 'user' as const, text: text.trim() },
            response,
          ].slice(-40),
        );
        return response;
      } catch (e) {
        setNotice(errorText(e));
        setQuestion(text);
        return null;
      } finally {
        mentorRef.current = false;
        setMentorBusy(false);
      }
    },
    [mode],
  );
  const browse = (path = 'all') => {
    setTrack(path);
    setFilter('all');
    setSearch('');
    setView('labs');
  };
  const updatePreferences = useCallback((changes: Partial<Profile>) => {
    const task = settingsWrites.current
      .catch(() => {})
      .then(() => api<Profile>('/api/settings', changes));
    settingsWrites.current = task;
    return task;
  }, []);
  const changeMode = (value: 'grounded' | 'experimental') => {
    const revision = ++modeRevision.current;
    setMode(value);
    void updatePreferences({ mentor_mode: value })
      .then((profile) => {
        persistedMode.current = profile.mentor_mode;
        setData((b) =>
          b
            ? {
                ...b,
                profile: { ...b.profile, mentor_mode: profile.mentor_mode },
              }
            : b,
        );
        if (revision === modeRevision.current) setMode(profile.mentor_mode);
      })
      .catch((error) => {
        if (revision === modeRevision.current) {
          setMode(persistedMode.current);
          setNotice(errorText(error));
        }
      });
  };
  const agentActions = useRef({ run, loadMission });
  useEffect(() => {
    agentActions.current = { run, loadMission };
  }, [run, loadMission]);
  useEffect(() => {
    type Tool = {
      name: string;
      title: string;
      description: string;
      inputSchema: object;
      annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
      execute: (input: unknown) => unknown;
    };
    const context = (
      document as Document & {
        modelContext?: {
          registerTool: (
            tool: Tool,
            options: { signal: AbortSignal },
          ) => void | Promise<void>;
        };
      }
    ).modelContext;
    if (!context?.registerTool) return;
    const lifetime = new AbortController();
    const ready = () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      );
    const empty = (input: unknown) => {
      if (
        !input ||
        typeof input !== 'object' ||
        Array.isArray(input) ||
        Object.keys(input).length
      )
        throw new Error('Expected an empty object.');
    };
    const tools: Tool[] = [
      {
        name: 'read_learning_state',
        title: 'Read learning progress',
        description:
          'Read the visible mission, total XP, completed mission IDs, and mission catalog. No code execution.',
        inputSchema: {
          type: 'object',
          properties: {},
          additionalProperties: false,
        },
        annotations: { readOnlyHint: true, untrustedContentHint: false },
        execute(input) {
          empty(input);
          return {
            mission: detailRef.current?.challenge.id || null,
            xp: dataRef.current?.profile.xp || 0,
            completed: Object.keys(dataRef.current?.profile.completed || {}),
            missions:
              dataRef.current?.challenges.map((c) => ({
                id: c.id,
                title: c.title,
              })) || [],
          };
        },
      },
      {
        name: 'open_mission',
        title: 'Open a C++ mission',
        description:
          'Save the current draft and open a mission in the visible editor. Use a mission ID from the catalog.',
        inputSchema: {
          type: 'object',
          properties: { mission_id: { type: 'string' } },
          required: ['mission_id'],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        async execute(input) {
          if (!input || typeof input !== 'object' || Array.isArray(input))
            throw new Error('Expected a mission_id object.');
          const x = input as Record<string, unknown>;
          if (
            Object.keys(x).length !== 1 ||
            typeof x.mission_id !== 'string' ||
            !dataRef.current?.challenges.some((c) => c.id === x.mission_id)
          )
            throw new Error('Unknown mission ID.');
          const result = await agentActions.current.loadMission(x.mission_id);
          if (!result)
            throw new Error('Mission did not open. Check the visible notice.');
          await ready();
          return {
            mission_id: result.challenge.id,
            title: result.challenge.title,
          };
        },
      },
      {
        name: 'run_current_code',
        title: 'Run current C++ examples',
        description:
          'Compile and execute the current editor code against public examples. Saves a practice attempt; does not submit for XP.',
        inputSchema: {
          type: 'object',
          properties: {},
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        async execute(input) {
          empty(input);
          const result = await agentActions.current.run('run');
          if (!result)
            throw new Error(
              'Run could not complete. Check the visible notice.',
            );
          await ready();
          return {
            compile_status: result.compile_status,
            passed: result.passed,
            total: result.total,
          };
        },
      },
    ];
    for (const tool of tools) {
      try {
        void Promise.resolve(
          context.registerTool(tool, { signal: lifetime.signal }),
        ).catch(() => {});
      } catch {
        /* Browsers without a usable registry retain every visible control. */
      }
    }
    return () => lifetime.abort();
  }, []);
  const p = data?.profile;
  const challenge = detail?.challenge;
  const currentNav = nav.find((n) => n.id === view)!;
  return (
    <SidebarProvider
      style={{ '--sidebar-width': '218px' } as React.CSSProperties}
    >
      <Sidebar className="quest-sidebar">
        <SidebarHeader>
          <button
            className="brand"
            onClick={() => setView('dashboard')}
            aria-label="Questline dashboard"
          >
            <span className="brand-symbol">
              <CodeXml size={23} />
            </span>
            questline<span className="brand-dot">.</span>
          </button>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>YOUR WORKSPACE</SidebarGroupLabel>
            <SidebarMenu>
              {nav.map((item) => (
                <SidebarMenuItem key={item.id}>
                  <SidebarMenuButton
                    isActive={view === item.id}
                    onClick={() => {
                      if (item.id === 'labs') browse();
                      else setView(item.id);
                    }}
                  >
                    <item.icon size={19} />
                    <span>{item.label}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroup>
          <SidebarGroup>
            <SidebarGroupLabel>LEARNING PATHS</SidebarGroupLabel>
            <SidebarMenu>
              {(Object.keys(trackNames) as TrackId[]).map((id) => {
                const Icon = trackIcons[id];
                return (
                  <SidebarMenuItem key={id}>
                    <SidebarMenuButton onClick={() => browse(id)}>
                      <Icon size={18} />
                      <span>{trackNames[id]}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroup>
          <div className="daily-card">
            <div>
              <Flame size={19} />
              <b>A little, every day</b>
            </div>
            <p>Make progress with {p?.daily_goal || 3} missions a day.</p>
            <Progress
              value={
                p ? Math.min(100, (p.today_completed / p.daily_goal) * 100) : 0
              }
            />
            <small>
              {p?.today_completed || 0} / {p?.daily_goal || 3} missions
              completed
            </small>
          </div>
        </SidebarContent>
        <SidebarFooter>
          <button
            className="profile-mini"
            onClick={() => setSettings(true)}
            disabled={!p}
            aria-label="Edit learning preferences"
          >
            <span>{(p?.name || 'D')[0].toUpperCase()}</span>
            <div>
              <strong>{p?.name || 'Developer'}</strong>
              <small>Level {p?.level || 1} · Explorer</small>
            </div>
            <Settings2 size={16} />
          </button>
        </SidebarFooter>
      </Sidebar>
      <main className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <SidebarTrigger />
            <span>{currentNav.label}</span>
            {view === 'practice' && challenge && (
              <>
                <ChevronRight size={14} />
                <strong>{trackNames[challenge.track]}</strong>
              </>
            )}
          </div>
          <div className="top-stats">
            <span className="streak">
              <Flame size={18} />
              {p?.streak || 0} day streak
            </span>
            <span className="xp-pill">
              <Zap size={17} />
              {p?.xp || 0} XP
            </span>
            <button
              className="avatar"
              aria-label="Learning preferences"
              onClick={() => setSettings(true)}
              disabled={!p}
            >
              {(p?.name || 'D')[0].toUpperCase()}
            </button>
          </div>
        </header>
        {notice && (
          <div className="notice" role="alert">
            <span>{notice}</span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setNotice('')}
              aria-label="Dismiss notice"
            >
              <X size={16} />
            </Button>
          </div>
        )}
        {loading || !data ? (
          <div className="connection-state">
            <span className="mentor-orb">
              {loading ? <LoaderCircle className="spin" /> : <CodeXml />}
            </span>
            <h1>
              {loading
                ? 'Opening your practice lab…'
                : 'Let’s reconnect your lab.'}
            </h1>
            <p>
              {loading
                ? 'Loading your missions and local compiler.'
                : 'Start the local server, then reconnect to restore your saved work.'}
            </p>
            {!loading && (
              <Button onClick={() => void connect()}>Reconnect</Button>
            )}
          </div>
        ) : (
          <>
            {view === 'dashboard' && (
              <Dashboard
                data={data}
                onOpen={(id) => void loadMission(id)}
                onBrowse={() => browse()}
              />
            )}{' '}
            {view === 'labs' && (
              <Catalog
                data={data}
                filter={filter}
                setFilter={setFilter}
                search={search}
                setSearch={setSearch}
                track={track}
                setTrack={setTrack}
                onOpen={(id) => void loadMission(id)}
              />
            )}{' '}
            {view === 'achievements' && (
              <Achievements data={data} onPractice={() => browse()} />
            )}{' '}
            {view === 'practice' && challenge && (
              <>
                <div className="practice-top">
                  <div>
                    <div className="eyebrow">
                      SMALL CHALLENGES. REAL SKILLS.
                    </div>
                    <h1>Let’s build your next skill.</h1>
                  </div>
                  <div className="practice-actions">
                    <Button variant="outline" onClick={() => setPicker(true)}>
                      <BookOpen size={16} />
                      Change mission
                    </Button>
                    <div className="level-widget">
                      <span>
                        LEVEL {p!.level} <b>EXPLORER</b>
                      </span>
                      <Progress value={(p!.level_xp / p!.level_target) * 100} />
                      <small>
                        {p!.level_xp} / {p!.level_target} XP to level{' '}
                        {p!.level + 1}
                      </small>
                    </div>
                  </div>
                </div>
                {!data.compiler.available && (
                  <div className="engine-warning">
                    <ShieldCheck size={18} />
                    <span>
                      The restricted C++ engine is unavailable. Install Apple
                      Command Line Tools with{' '}
                      <code>xcode-select --install</code>, then restart the app.
                    </span>
                  </div>
                )}
                <div
                  className={'workspace ' + (switching ? 'switching' : '')}
                  aria-busy={switching}
                >
                  <ProblemPanel
                    challenge={challenge}
                    onBrowse={() => browse(challenge.track)}
                  />
                  <section className="editor-panel">
                    <div className="editor-header">
                      <span>
                        <CodeXml size={17} />
                        <b>main.cpp</b>
                      </span>
                      <div>
                        <span>C++20</span>
                        <Button
                          variant="ghost"
                          size="icon"
                          aria-label="Restore starter code"
                          title="Restore starter code (Undo recovers your edits)"
                          onClick={() => setCode(challenge.starter_code)}
                          disabled={running || switching}
                        >
                          <RotateCcw size={16} />
                        </Button>
                      </div>
                    </div>
                    <CodeEditor
                      key={challenge.id}
                      readOnly={running || switching}
                      value={code}
                      onChange={setCode}
                      onRun={() =>
                        void run(inputMode === 'custom' ? 'custom' : 'run')
                      }
                    />
                    <div className="editor-status">
                      <span>
                        <Circle size={7} fill="currentColor" />
                        {saveStatus}
                      </span>
                      <span>⌘ / Ctrl + Enter to run</span>
                    </div>
                    <div className="run-toolbar">
                      <Select
                        value={inputMode}
                        onValueChange={(v) => setInputMode(String(v))}
                        disabled={running}
                      >
                        <SelectTrigger aria-label="Input mode">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="samples">Examples</SelectItem>
                          <SelectItem value="custom">Custom input</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button
                        className="run-button"
                        onClick={() =>
                          void run(inputMode === 'custom' ? 'custom' : 'run')
                        }
                        disabled={
                          running || switching || !data.compiler.available
                        }
                      >
                        {running ? (
                          <LoaderCircle className="spin" size={15} />
                        ) : (
                          <Play size={15} />
                        )}
                        Run code
                      </Button>
                      <Button
                        className="submit-button"
                        onClick={() => void run('submit')}
                        disabled={
                          running || switching || !data.compiler.available
                        }
                      >
                        Submit
                        <ArrowUpRight size={15} />
                      </Button>
                    </div>
                    {inputMode === 'custom' && (
                      <div className="custom-input">
                        <Label htmlFor="program-input">
                          Program input (stdin)
                        </Label>
                        <Textarea
                          id="program-input"
                          value={stdin}
                          onChange={(e) => setStdin(e.target.value)}
                          maxLength={16000}
                          spellCheck={false}
                        />
                        <small>
                          Custom runs show output. Submit checks all{' '}
                          {challenge.tests.length + challenge.hidden_count}{' '}
                          mission tests.
                        </small>
                      </div>
                    )}
                    <ConsolePanel
                      stale={resultStale}
                      result={result}
                      running={running}
                      onDebug={() =>
                        void ask(
                          'Help me understand my last compiler or test result.',
                        )
                      }
                    />
                  </section>
                  <MentorPanel
                    challenge={challenge}
                    name={p!.name}
                    messages={messages}
                    question={question}
                    setQuestion={setQuestion}
                    onAsk={(s) => void ask(s)}
                    busy={mentorBusy}
                    mode={mode}
                    onMode={changeMode}
                    experimentalAvailable={data.custom_model.available}
                  />
                </div>
                <footer className="workspace-footer">
                  <span>
                    <Leaf size={14} />
                    Learning by doing, one mission at a time.
                  </span>
                  <span>
                    <ShieldCheck size={14} />
                    Local C++20 · {challenge.tests.length} examples +{' '}
                    {challenge.hidden_count} hidden tests
                  </span>
                </footer>
              </>
            )}
          </>
        )}
        <Dialog open={picker} onOpenChange={setPicker}>
          <DialogContent className="mission-dialog">
            <DialogHeader>
              <DialogTitle>Choose your next mission</DialogTitle>
              <DialogDescription>
                Explore {data?.challenges.length || 24} real-world problems,
                from first steps to advanced algorithms.
              </DialogDescription>
            </DialogHeader>
            <div className="search-field">
              <Search size={17} />
              <Input
                value={pickerSearch}
                onChange={(e) => setPickerSearch(e.target.value)}
                placeholder="Search missions or skills"
                aria-label="Find a mission"
              />
            </div>
            <div className="mission-picker-list">
              {data?.challenges
                .filter((c) =>
                  (c.title + ' ' + c.tags.join(' '))
                    .toLowerCase()
                    .includes(pickerSearch.toLowerCase()),
                )
                .map((c) => (
                  <button
                    key={c.id}
                    onClick={() => void loadMission(c.id)}
                    disabled={running || mentorBusy || switching}
                  >
                    <span className={'difficulty ' + c.difficulty}>
                      {c.difficulty}
                    </span>
                    <div>
                      <strong>{c.title}</strong>
                      <small>
                        {trackNames[c.track]} · {c.xp} XP
                      </small>
                    </div>
                    {p?.completed[c.id] ? (
                      <Check size={17} />
                    ) : (
                      <ChevronRight size={17} />
                    )}
                  </button>
                ))}
              {data &&
                !data.challenges.some((c) =>
                  (c.title + ' ' + c.tags.join(' '))
                    .toLowerCase()
                    .includes(pickerSearch.toLowerCase()),
                ) && (
                  <p className="picker-empty">
                    No matching missions. Try “game”, “memory”, or “input”.
                  </p>
                )}
            </div>
          </DialogContent>
        </Dialog>
        <Dialog open={settings} onOpenChange={setSettings}>
          <DialogContent className="settings-dialog">
            <DialogHeader>
              <DialogTitle>Make this your learning space</DialogTitle>
              <DialogDescription>
                Your preferences and progress stay on this Mac.
              </DialogDescription>
            </DialogHeader>
            {p && (
              <Preferences
                key={settings ? 'open' : 'closed'}
                profile={p}
                onSave={async (changes) => {
                  const profile = await updatePreferences(changes);
                  setData((b) =>
                    b
                      ? {
                          ...b,
                          profile: {
                            ...b.profile,
                            name: profile.name,
                            difficulty: profile.difficulty,
                            track: profile.track,
                            daily_goal: profile.daily_goal,
                          },
                        }
                      : b,
                  );
                  setSettings(false);
                }}
              />
            )}
          </DialogContent>
        </Dialog>
        <Dialog
          open={award > 0}
          onOpenChange={(open) => {
            if (!open) setAward(0);
          }}
        >
          <DialogContent className="award-dialog">
            <DialogHeader>
              <span className="earned-trophy">
                <Trophy size={36} />
              </span>
              <DialogTitle>That’s a skill worth keeping.</DialogTitle>
              <DialogDescription>
                All tests passed. You worked through a real problem and made
                your code deliver.
              </DialogDescription>
            </DialogHeader>
            <div className="award-xp">
              <Zap size={24} />+{award} XP
            </div>
            <p>
              {p?.xp || 0} total XP · Level {p?.level || 1}
            </p>
            <Button
              onClick={() => {
                setAward(0);
                const next =
                  data?.challenges.find(
                    (c) => !p?.completed[c.id] && c.track === challenge?.track,
                  ) || data?.challenges.find((c) => !p?.completed[c.id]);
                if (next) void loadMission(next.id);
                else setView('achievements');
              }}
            >
              Keep the momentum
              <ArrowRight size={17} />
            </Button>
            <Button variant="ghost" onClick={() => setAward(0)}>
              Stay with this mission
            </Button>
          </DialogContent>
        </Dialog>
      </main>
    </SidebarProvider>
  );
}
function Preferences({
  profile,
  onSave,
}: {
  profile: Profile;
  onSave: (changes: Partial<Profile>) => Promise<void>;
}) {
  const [name, setName] = useState(profile.name);
  const [goal, setGoal] = useState(profile.daily_goal);
  const [level, setLevel] = useState(profile.difficulty);
  const [path, setPath] = useState(profile.track);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  return (
    <form
      className="preferences-form"
      onSubmit={async (e) => {
        e.preventDefault();
        setSaving(true);
        setError('');
        try {
          await onSave({
            name: name.trim(),
            daily_goal: goal,
            difficulty: level,
            track: path,
          });
        } catch (e) {
          setError(errorText(e));
        } finally {
          setSaving(false);
        }
      }}
    >
      <Label htmlFor="learner-name">What should we call you?</Label>
      <Input
        id="learner-name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        required
        maxLength={32}
      />
      <Label htmlFor="daily-goal">Daily mission goal</Label>
      <Input
        id="daily-goal"
        type="number"
        min={1}
        max={10}
        value={goal}
        onChange={(e) => setGoal(Number(e.target.value))}
        required
      />
      <Label htmlFor="starting-level">Starting level</Label>
      <Select
        value={level}
        onValueChange={(v) => setLevel(v as Profile['difficulty'])}
      >
        <SelectTrigger id="starting-level" aria-label="Starting level">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {['beginner', 'intermediate', 'advanced', 'all'].map((v) => (
            <SelectItem value={v} key={v}>
              {v === 'all'
                ? 'Explore every level'
                : v[0].toUpperCase() + v.slice(1)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Label htmlFor="learning-focus">Your focus</Label>
      <Select
        value={path}
        onValueChange={(v) => setPath(v as Profile['track'])}
      >
        <SelectTrigger id="learning-focus" aria-label="Your focus">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">A little of everything</SelectItem>
          {(Object.keys(trackNames) as TrackId[]).map((id) => (
            <SelectItem key={id} value={id}>
              {trackNames[id]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {error && (
        <p role="alert" className="form-error">
          {error}
        </p>
      )}
      <Button type="submit" disabled={saving || !name.trim()}>
        {saving ? (
          <LoaderCircle className="spin" size={16} />
        ) : (
          <Check size={16} />
        )}
        Save preferences
      </Button>
    </form>
  );
}
