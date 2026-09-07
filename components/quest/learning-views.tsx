'use client';
import {
  ArrowUpRight,
  Check,
  Cpu,
  Flame,
  Layers,
  Leaf,
  Search,
  ShieldCheck,
  Swords,
  Target,
  Trophy,
  Zap,
  type LucideIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type {
  Bootstrap,
  ChallengeSummary,
  Profile,
  TrackId,
} from '@/lib/quest-types';
export const trackIcons: Record<TrackId, LucideIcon> = {
  foundations: Leaf,
  systems: Cpu,
  games: Swords,
  compilers: Layers,
};
export const trackNames: Record<TrackId, string> = {
  foundations: 'Foundations',
  systems: 'Systems & memory',
  games: 'Game development',
  compilers: 'Compiler workshop',
};
export function MissionCard({
  mission,
  profile,
  onOpen,
}: {
  mission: ChallengeSummary;
  profile: Profile;
  onOpen: (id: string) => void;
}) {
  const Icon = trackIcons[mission.track];
  const solved = Boolean(profile.completed[mission.id]);
  return (
    <button
      className={
        'mission-card track-' + mission.track + (solved ? ' solved' : '')
      }
      onClick={() => onOpen(mission.id)}
    >
      <div className="card-top">
        <span className="track-icon">
          <Icon size={23} />
        </span>
        <span className={'difficulty ' + mission.difficulty}>
          {mission.difficulty}
        </span>
      </div>
      <div className="card-track">{trackNames[mission.track]}</div>
      <h3>{mission.title}</h3>
      <p>{mission.summary}</p>
      <div className="card-bottom">
        <span>
          {mission.minutes} min <i /> <Zap size={14} />
          {mission.xp} XP
        </span>
        <b>
          {solved ? (
            <>
              <Check size={14} />
              Completed
            </>
          ) : (
            <>
              Start mission
              <ArrowUpRight size={15} />
            </>
          )}
        </b>
      </div>
    </button>
  );
}
export function Dashboard({
  data,
  onOpen,
  onBrowse,
}: {
  data: Bootstrap;
  onOpen: (id: string) => void;
  onBrowse: () => void;
}) {
  const p = data.profile;
  const next =
    data.challenges.find(
      (c) =>
        !p.completed[c.id] &&
        (p.difficulty === 'all' || c.difficulty === p.difficulty) &&
        (p.track === 'all' || c.track === p.track),
    ) ||
    data.challenges.find((c) => !p.completed[c.id]) ||
    data.challenges[0];
  return (
    <div className="overview-view">
      <div className="view-heading">
        <div className="eyebrow">YOUR LEARNING JOURNEY</div>
        <h1>Good to see you, {p.name}.</h1>
        <p>Small wins today. Real skills for tomorrow.</p>
      </div>
      <div className="stats-grid">
        {[
          {
            label: 'Total XP',
            value: p.xp,
            icon: Zap,
            caption: `Level ${p.level} explorer`,
          },
          {
            label: 'Missions completed',
            value: Object.keys(p.completed).length,
            icon: ShieldCheck,
            caption: `${data.challenges.length} missions to explore`,
          },
          {
            label: 'Current streak',
            value: p.streak + ' days',
            icon: Flame,
            caption: 'A little practice goes a long way',
          },
          {
            label: 'Today’s goal',
            value:
              Math.min(p.today_completed, p.daily_goal) + ' / ' + p.daily_goal,
            icon: Target,
            caption: 'New missions completed today',
          },
        ].map((stat) => (
          <section className="stat-card" key={stat.label}>
            <stat.icon size={21} />
            <span>{stat.label}</span>
            <strong>{stat.value}</strong>
            <small>{stat.caption}</small>
          </section>
        ))}
      </div>
      <div className="dashboard-middle">
        <section className="continue-card">
          <span className="eyebrow">YOUR NEXT MISSION</span>
          <h2>{next.title}</h2>
          <p>{next.summary}</p>
          <div className="continue-meta">
            <span className={'difficulty ' + next.difficulty}>
              {next.difficulty}
            </span>
            <span>
              <Zap size={15} />
              {next.xp} XP
            </span>
            <span>{next.minutes} min</span>
          </div>
          <Button onClick={() => onOpen(next.id)}>
            Let’s practice
            <ArrowUpRight size={16} />
          </Button>
        </section>
        <section className="activity-card">
          <h3>Your week in progress</h3>
          <p>Every attempt counts as practice.</p>
          <div className="activity-bars">
            {p.activity.map((day) => (
              <div
                key={day.date}
                title={`${day.attempts} attempts on ${day.date}`}
              >
                <b>{day.attempts || ''}</b>
                <span
                  style={{
                    height: Math.max(5, Math.min(90, day.attempts * 15)),
                  }}
                />
                <small>{day.label}</small>
              </div>
            ))}
          </div>
        </section>
      </div>
      <div className="section-heading">
        <div>
          <h2>Pick your path</h2>
          <p>Start with the basics or jump into a real-world problem.</p>
        </div>
        <Button variant="ghost" onClick={onBrowse}>
          Explore all
          <ArrowUpRight size={16} />
        </Button>
      </div>
      <div className="track-grid">
        {data.tracks.map((track) => {
          const Icon = trackIcons[track.id];
          const missions = data.challenges.filter((c) => c.track === track.id);
          const solved = missions.filter((c) => p.completed[c.id]).length;
          return (
            <button
              key={track.id}
              className={'path-card track-' + track.id}
              onClick={() =>
                onOpen(
                  missions.find((c) => !p.completed[c.id])?.id ||
                    missions[0].id,
                )
              }
            >
              <span className="track-icon">
                <Icon size={24} />
              </span>
              <h3>{track.title}</h3>
              <p>{track.description}</p>
              <Progress value={(solved / missions.length) * 100} />
              <span className="path-bottom">
                {solved} of {missions.length} missions
                <ArrowUpRight size={17} />
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
export function Catalog({
  data,
  filter,
  setFilter,
  search,
  setSearch,
  track,
  setTrack,
  onOpen,
}: {
  data: Bootstrap;
  filter: string;
  setFilter: (s: string) => void;
  search: string;
  setSearch: (s: string) => void;
  track: string;
  setTrack: (s: string) => void;
  onOpen: (id: string) => void;
}) {
  const missions = data.challenges.filter(
    (c) =>
      (filter === 'all' || c.difficulty === filter) &&
      (track === 'all' || c.track === track) &&
      `${c.title} ${c.summary} ${c.tags.join(' ')}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  return (
    <div className="overview-view">
      <div className="view-heading">
        <div className="eyebrow">LEARN IT. BUILD IT. UNDERSTAND IT.</div>
        <h1>Real-world labs</h1>
        <p>
          From your first variable to caches, game loops, and compiler
          algorithms.
        </p>
      </div>
      <div className="catalog-controls">
        <Tabs value={filter} onValueChange={(v) => setFilter(String(v))}>
          <TabsList className="difficulty-tabs">
            {['all', 'beginner', 'intermediate', 'advanced'].map((value) => (
              <TabsTrigger value={value} key={value}>
                {value === 'all'
                  ? 'All levels'
                  : value[0].toUpperCase() + value.slice(1)}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <Select value={track} onValueChange={(v) => setTrack(String(v))}>
          <SelectTrigger aria-label="Learning path">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {[{ id: 'all', title: 'All paths' }, ...data.tracks].map((t) => (
              <SelectItem key={t.id} value={t.id}>
                {t.title}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="search-field">
          <Search size={17} />
          <Input
            aria-label="Search missions"
            placeholder="Find a skill or mission"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>
      <div className="catalog-caption">
        <span>{missions.length} missions to explore</span>
        <span>
          <ShieldCheck size={15} />
          Real tests. Real progress.
        </span>
      </div>
      <div className="mission-grid">
        {missions.map((m) => (
          <MissionCard
            key={m.id}
            mission={m}
            profile={data.profile}
            onOpen={onOpen}
          />
        ))}
      </div>
      {!missions.length && (
        <div className="empty-state">
          <Search size={30} />
          <h3>No missions match that search.</h3>
          <p>Try another skill, level, or path.</p>
          <Button
            variant="outline"
            onClick={() => {
              setSearch('');
              setTrack('all');
              setFilter('all');
            }}
          >
            Clear filters
          </Button>
        </div>
      )}
    </div>
  );
}
export function Achievements({
  data,
  onPractice,
}: {
  data: Bootstrap;
  onPractice: () => void;
}) {
  const p = data.profile;
  const completed = data.challenges.filter((c) => p.completed[c.id]);
  const badges = [
    {
      id: 'first',
      name: 'First spark',
      description: 'Complete your first mission.',
      icon: Zap,
      value: completed.length,
      target: 1,
    },
    {
      id: 'five',
      name: 'Problem solver',
      description: 'Complete five different missions.',
      icon: ShieldCheck,
      value: completed.length,
      target: 5,
    },
    {
      id: 'advanced',
      name: 'Beyond the basics',
      description: 'Solve one advanced mission.',
      icon: Swords,
      value: completed.filter((c) => c.difficulty === 'advanced').length,
      target: 1,
    },
    {
      id: 'compiler',
      name: 'Compiler explorer',
      description: 'Complete three compiler missions.',
      icon: Cpu,
      value: completed.filter((c) => c.track === 'compilers').length,
      target: 3,
    },
    {
      id: 'paths',
      name: 'Four corners',
      description: 'Solve a mission in every path.',
      icon: Layers,
      value: new Set(completed.map((c) => c.track)).size,
      target: 4,
    },
    {
      id: 'all',
      name: 'Quest master',
      description: 'Complete the entire collection.',
      icon: Trophy,
      value: completed.length,
      target: data.challenges.length,
    },
  ];
  return (
    <div className="overview-view">
      <div className="view-heading">
        <div className="eyebrow">PROOF OF YOUR PRACTICE</div>
        <h1>Your wins, collected.</h1>
        <p>
          Badges are earned by solving real challenges. Every one tells a story.
        </p>
      </div>
      <div className="achievement-summary">
        <Trophy size={32} />
        <div>
          <strong>
            {badges.filter((b) => b.value >= b.target).length} of{' '}
            {badges.length} badges earned
          </strong>
          <p>
            {p.xp} XP earned across {completed.length} missions
          </p>
        </div>
        <Button onClick={onPractice}>
          Keep exploring
          <ArrowUpRight size={16} />
        </Button>
      </div>
      <div className="badge-grid">
        {badges.map((b) => {
          const earned = b.value >= b.target;
          return (
            <section
              key={b.id}
              className={'badge-card ' + (earned ? 'earned' : 'locked')}
            >
              <span className="badge-emblem">
                <b.icon size={34} />
              </span>
              <small>{earned ? 'UNLOCKED' : 'KEEP GOING'}</small>
              <h3>{b.name}</h3>
              <p>{b.description}</p>
              <Progress value={Math.min(100, (b.value / b.target) * 100)} />
              <span>
                {Math.min(b.value, b.target)} / {b.target}
              </span>
            </section>
          );
        })}
      </div>
    </div>
  );
}
