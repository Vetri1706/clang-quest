export type Difficulty = 'beginner' | 'intermediate' | 'advanced';
export type TrackId = 'foundations' | 'systems' | 'games' | 'compilers';
export type ViewName = 'practice' | 'dashboard' | 'labs' | 'achievements';
export interface ChallengeSummary {
  id: string;
  title: string;
  summary: string;
  difficulty: Difficulty;
  track: TrackId;
  order: number;
  xp: number;
  minutes: number;
  tags: string[];
}
export interface Challenge extends ChallengeSummary {
  story: string;
  prompt: string;
  input_format: string;
  output_format: string;
  constraints: string[];
  starter_code: string;
  concepts: { title: string; body: string; code?: string }[];
  objectives: string[];
  tests: { input: string; output: string; hidden: boolean }[];
  hidden_count: number;
}
export interface Profile {
  name: string;
  difficulty: Difficulty | 'all';
  track: TrackId | 'all';
  daily_goal: number;
  mentor_mode: 'grounded' | 'experimental';
  completed: Record<
    string,
    { challenge_id: string; xp: number; completed_at: string }
  >;
  xp: number;
  level: number;
  level_xp: number;
  level_target: number;
  streak: number;
  today_completed: number;
  total_attempts: number;
  activity: {
    date: string;
    label: string;
    attempts: number;
    completed: number;
  }[];
  last_challenge: string;
}
export interface CaseResult {
  index: number;
  hidden: boolean;
  passed: boolean;
  status: string;
  time_ms?: number;
  input?: string;
  expected?: string;
  stdout?: string;
  stderr?: string;
  actual?: string;
}
export interface RunResult {
  source_sha256?: string;
  compile_status: string;
  diagnostics: string;
  duration_ms: number;
  results: CaseResult[];
  mode: 'run' | 'submit' | 'custom';
  passed: number;
  total: number;
  xp_awarded?: number;
  profile?: Profile;
}
export interface Message {
  id?: number;
  role: 'user' | 'assistant';
  text: string;
  kind?: string;
  sources?: { title: string; url: string }[];
  suggestions?: string[];
  model?: { mode: string; label: string };
  next_hint_level?: number;
}
export interface Detail {
  challenge: Challenge;
  source: string;
  draft_updated_at: string | null;
  messages: Message[];
  hint_level: number;
  last_run: RunResult | null;
}
export interface Bootstrap {
  csrf: string;
  tracks: { id: TrackId; title: string; description: string; color: string }[];
  challenges: ChallengeSummary[];
  profile: Profile;
  compiler: {
    available: boolean;
    engine?: string;
    limits?: Record<string, number>;
  };
  custom_model: {
    available: boolean;
    name: string;
    quality_gate: string;
    trained_parameters: number;
  };
}
