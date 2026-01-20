import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Cpu, Gauge } from 'lucide-react';
import { Link } from 'react-router-dom';
import { RaceData } from '@/types/race';
import { useRaceLoop } from '@/hooks/useRaceLoop';
import { RaceCanvas } from '@/components/RaceCanvas';
import { PlaybackControls } from '@/components/PlaybackControls';
import { TelemetryCockpit } from '@/components/TelemetryCockpit';
import { ComparisonCharts } from '@/components/ComparisonCharts';
import { AgentBadges } from '@/components/AgentBadges';
import { ResearchCharts } from '@/components/ResearchCharts';
import { ExploitationFinding } from '@/components/ExploitationFinding';
import { GeneralizationFinding } from '@/components/GeneralizationFinding';
import { Footer } from '@/components/Footer';

const Index = () => {
  // Scenario state removed for Unified Master Simulation
  const [rlData, setRlData] = useState<RaceData | null>(null);
  const [esData, setEsData] = useState<RaceData | null>(null);
  const [robustnessData, setRobustnessData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTrack, setActiveTrack] = useState<'oval' | 'fig8' | 'random1' | 'random2' | 'random3'>('oval');

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const timestamp = Date.now();
        console.log(`Loading Data for ${activeTrack}...`);

        // Always load Master Simulation (Figure 8 Real)
        const rlUrl = `/rl_real_fig8.json?t=${timestamp}`;
        const esUrl = `/es_real_fig8.json?t=${timestamp}`;

        const [rlResponse, esResponse, robustResponse] = await Promise.all([
          fetch(rlUrl),
          fetch(esUrl),
          fetch(`/exp_robustness.json?t=${timestamp}`).catch(() => ({ ok: false, json: () => null })),
        ]);

        let rl = null;
        let es = null;

        if (rlResponse.ok) rl = await rlResponse.json();
        if (esResponse.ok) es = await esResponse.json();

        let robust = null;
        try {
          if (robustResponse.ok) robust = await robustResponse.json();
        } catch (e) { console.warn("No research data found"); }

        setRlData(rl);
        setEsData(es);
        setRobustnessData(robust);
      } catch (error) {
        console.error('Failed to load race data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [activeTrack]);

  const {
    isPlaying,
    playbackSpeed,
    setPlaybackSpeed,
    loop,
    setLoop,
    currentTime,
    maxTime,
    rlState,
    esState,
    rlTrail,
    esTrail,
    play,
    pause,
    reset,
    seek,
  } = useRaceLoop(rlData, esData);

  if (loading) {
    return (
      <div className="min-h-screen bg-background grid-bg flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center"
        >
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-primary font-mono animate-pulse">LOADING SIMULATION DATA...</p>
        </motion.div>
      </div>
    );
  }

  const trackPoints = rlData?.metadata.track_points || esData?.metadata.track_points || [];

  return (
    <div className="min-h-screen bg-background grid-bg relative overflow-hidden">
      {/* Scanline overlay */}
      <div className="fixed inset-0 scanline pointer-events-none z-50" />

      {/* Header */}
      <header className="relative z-10 border-b border-muted bg-card/50 backdrop-blur-md">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            {/* Title Section */}
            <motion.div
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="flex items-center gap-3"
            >
              <div className="p-2 rounded-lg bg-primary/20 cyber-glow">
                <Cpu className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-lg sm:text-xl font-display font-bold text-foreground tracking-wider">
                  POLICY UNDER <span className="text-primary neon-text">PRESSURE</span>
                </h1>
                <p className="text-xs text-muted-foreground font-mono max-w-md sm:max-w-none">
                  Interactive experiments exploring how intelligent agents behave when rewards, sensors, and physics turn hostile
                </p>
              </div>
            </motion.div>

            {/* NAVIGATION LINKS */}
            <div className="flex items-center gap-1 sm:gap-2 bg-background/50 p-1 rounded-lg border border-muted w-full sm:w-auto overflow-x-auto">
              <Link
                to="/segment1"
                className="px-2 sm:px-4 py-2 text-xs font-mono font-bold rounded-md bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 transition-all flex items-center gap-1 sm:gap-2 whitespace-nowrap flex-1 sm:flex-none justify-center"
              >
                <Activity className="w-3 h-3" />
                <span className="hidden xs:inline">SEGMENT 1</span>
                <span className="xs:hidden">S1</span>
              </Link>
              <Link
                to="/segment2"
                className="px-2 sm:px-4 py-2 text-xs font-mono font-bold rounded-md bg-secondary/10 hover:bg-secondary/20 text-secondary border border-secondary/20 transition-all flex items-center gap-1 sm:gap-2 whitespace-nowrap flex-1 sm:flex-none justify-center"
              >
                <Activity className="w-3 h-3" />
                <span className="hidden xs:inline">SEGMENT 2</span>
                <span className="xs:hidden">S2</span>
              </Link>
              <Link
                to="/segment3"
                className="px-2 sm:px-4 py-2 text-xs font-mono font-bold rounded-md bg-white/10 hover:bg-white/20 text-white border border-white/20 transition-all flex items-center gap-1 sm:gap-2 whitespace-nowrap flex-1 sm:flex-none justify-center"
              >
                <Activity className="w-3 h-3" />
                <span className="hidden xs:inline">SEGMENT 3</span>
                <span className="xs:hidden">S3</span>
              </Link>
              <Link
                to="/segment4"
                className="px-2 sm:px-4 py-2 text-xs font-mono font-bold rounded-md bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 border border-amber-500/20 transition-all flex items-center gap-1 sm:gap-2 whitespace-nowrap flex-1 sm:flex-none justify-center"
              >
                <Activity className="w-3 h-3" />
                <span className="hidden xs:inline">SEGMENT 4</span>
                <span className="xs:hidden">S4</span>
              </Link>
            </div>

            {/* Status (No Buttons) */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              className="flex items-center gap-6"
            >
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-primary animate-pulse" />
                <span className="text-xs font-mono text-muted-foreground">LIVE</span>
              </div>
              <div className="flex items-center gap-4 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-rl shadow-lg shadow-rl/50 animate-pulse" />
                  <span className="text-rl animate-pulse">RL</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-es shadow-lg shadow-es/50 animate-pulse" />
                  <span className="text-es animate-pulse">ES</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Stats & Badges */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="space-y-6"
          >
            {/* Quick Stats */}
            <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted p-4">
              <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-4">
                Race Statistics
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-background/50 rounded-lg p-3 border border-muted">
                  <div className="text-[10px] text-muted-foreground mb-1">RL TIME</div>
                  <div className="text-lg font-mono text-rl">
                    {rlData?.metadata.total_time.toFixed(1)}s
                  </div>
                </div>
                <div className="bg-background/50 rounded-lg p-3 border border-muted">
                  <div className="text-[10px] text-muted-foreground mb-1">ES TIME</div>
                  <div className="text-lg font-mono text-es">
                    {esData?.metadata.total_time.toFixed(1)}s
                  </div>
                </div>
                <div className="bg-background/50 rounded-lg p-3 border border-muted">
                  <div className="text-[10px] text-muted-foreground mb-1">RL REWARD</div>
                  <div className="text-lg font-mono text-rl">
                    {rlData?.metadata.total_reward.toFixed(1)}
                  </div>
                </div>
                <div className="bg-background/50 rounded-lg p-3 border border-muted">
                  <div className="text-[10px] text-muted-foreground mb-1">ES REWARD</div>
                  <div className="text-lg font-mono text-es">
                    {esData?.metadata.total_reward.toFixed(1)}
                  </div>
                </div>
              </div>
            </div>

            {/* Badges */}
            <AgentBadges rlData={rlData} esData={esData} />
          </motion.div>

          {/* Center - Race Canvas */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-1 space-y-4"
          >
            <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted p-4">
              <div className="flex items-center gap-2 mb-4">
                <Gauge className="w-4 h-4 text-track" />
                <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
                  Race Arena
                </h3>
              </div>
              <RaceCanvas
                trackPoints={trackPoints}
                rlState={rlState}
                esState={esState}
                rlTrail={rlTrail}
                esTrail={esTrail}
              />
            </div>

            {/* Playback Controls */}
            <PlaybackControls
              isPlaying={isPlaying}
              currentTime={currentTime}
              maxTime={maxTime}
              playbackSpeed={playbackSpeed}
              loop={loop}
              onPlay={play}
              onPause={pause}
              onReset={reset}
              onSeek={seek}
              onSpeedChange={setPlaybackSpeed}
              onLoopChange={setLoop}
            />
          </motion.div>

          {/* Right Panel - Telemetry & Charts */}
          <motion.div
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="space-y-6"
          >
            {/* Telemetry */}
            <TelemetryCockpit rlState={rlState} esState={esState} />

            {/* Charts */}
            <ComparisonCharts
              rlData={rlData}
              esData={esData}
              currentTime={currentTime}
            />
          </motion.div>
        </div>

        {/* Research Section */}
        <div className="mt-12 space-y-8">
          <div className="flex items-center gap-4">
            <div className="h-px bg-muted flex-1" />
            <h2 className="text-xl font-display font-bold text-foreground tracking-wider">
              RESEARCH <span className="text-primary neon-text">FINDINGS</span>
            </h2>
            <div className="h-px bg-muted flex-1" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Segment 1: Phase 2 Stress Tests (5 experiments) */}
            <Link to="/segment1" className="block group">
              <div className="bg-card/30 backdrop-blur-sm rounded-lg border border-primary/20 p-4 hover:bg-card/50 transition-all hover:border-primary/40">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-primary" />
                  <h3 className="text-sm font-mono font-bold text-primary">SEGMENT 1</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-3">Phase 2: Stress Testing (5 experiments)</p>
                <ul className="space-y-1 text-xs text-foreground/70">
                  <li>→ New World (Generalization)</li>
                  <li>→ Ice Patch (Physics Robustness)</li>
                  <li>→ Foggy Sensor (Noise)</li>
                  <li>→ Blindfold (Partial Observability)</li>
                  <li>→ Wake Up Call (Adversarial Starts)</li>
                </ul>
                <p className="text-xs text-primary/80 mt-3 font-mono group-hover:underline">View Details →</p>
              </div>
            </Link>

            {/* Segment 2: Quantitative Robustness (3 experiments) */}
            <Link to="/segment2" className="block group">
              <div className="bg-card/30 backdrop-blur-sm rounded-lg border border-secondary/20 p-4 hover:bg-card/50 transition-all hover:border-secondary/40">
                <div className="flex items-center gap-2 mb-2">
                  <Gauge className="w-4 h-4 text-secondary" />
                  <h3 className="text-sm font-mono font-bold text-secondary">SEGMENT 2</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-3">Quantitative Robustness (3 experiments)</p>
                <ul className="space-y-1 text-xs text-foreground/70">
                  <li>→ Friction Sweep (0.3-1.0)</li>
                  <li>→ Noise Gradient (0%-30%)</li>
                  <li>→ Masking Patterns (Velocity Dropout)</li>
                </ul>
                <p className="text-xs text-secondary/80 mt-3 font-mono group-hover:underline">View Details →</p>
              </div>
            </Link>

            {/* Segment 3: Gradient Analysis (3 experiments) */}
            <Link to="/segment3" className="block group">
              <div className="bg-card/30 backdrop-blur-sm rounded-lg border border-white/20 p-4 hover:bg-card/50 transition-all hover:border-white/40">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-white" />
                  <h3 className="text-sm font-mono font-bold text-white">SEGMENT 3</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-3">Gradient Analysis (3 experiments)</p>
                <ul className="space-y-1 text-xs text-foreground/70">
                  <li>→ Learning Dynamics (α Boost)</li>
                  <li>→ Phase Transitions (Training Phases)</li>
                  <li>→ Policy Space (Trajectory Patterns)</li>
                </ul>
                <p className="text-xs text-white/80 mt-3 font-mono group-hover:underline">View Details →</p>
              </div>
            </Link>

            {/* Segment 4: Reward Engineering (4 experiments) */}
            <Link to="/segment4" className="block group">
              <div className="bg-card/30 backdrop-blur-sm rounded-lg border border-amber-500/20 p-4 hover:bg-card/50 transition-all hover:border-amber-500/40">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-amber-500" />
                  <h3 className="text-sm font-mono font-bold text-amber-500">SEGMENT 4</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-3">Reward Engineering (4 experiments)</p>
                <ul className="space-y-1 text-xs text-foreground/70">
                  <li>→ Reward Exploitation</li>
                  <li>→ Reward Sensitivity</li>
                  <li>→ Alignment Test</li>
                  <li>→ Compute Scaling</li>
                </ul>
                <p className="text-xs text-amber-500/80 mt-3 font-mono group-hover:underline">View Details →</p>
              </div>
            </Link>

            {/* Summary Card */}
            <div className="bg-gradient-to-br from-primary/10 to-secondary/10 rounded-lg border border-primary/30 p-4">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-primary" />
                <h3 className="text-sm font-mono font-bold text-foreground">TOTAL EXPERIMENTS</h3>
              </div>
              <div className="text-3xl font-display font-bold text-primary mb-2">15</div>
              <p className="text-xs text-muted-foreground">
                Comprehensive behavioral analysis across stress testing, robustness quantification, gradient dynamics, and reward engineering.
              </p>
            </div>

            {/* Key Finding Card */}
            <div className="bg-card/20 backdrop-blur-sm rounded-lg border border-muted p-4">
              <h3 className="text-sm font-mono font-bold text-foreground mb-2">KEY INSIGHT</h3>
              <p className="text-xs text-foreground/70 leading-relaxed">
                <strong className="text-primary">RL</strong> optimizes for peak performance (fragile), while <strong className="text-secondary">ES</strong> optimizes for survival (robust).
                Neither algorithm dominates—trade-offs emerge across all 15 experiments.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer status bar */}
      <footer className="fixed bottom-0 left-0 right-0 bg-card/80 backdrop-blur-md border-t border-muted py-2 px-4 z-40">
        <div className="container mx-auto flex items-center justify-between text-[10px] font-mono text-muted-foreground">
          <div className="flex items-center gap-4">
            <span>FPS: 60</span>
            <span>|</span>
            <span>PLAYBACK: {playbackSpeed}x</span>
            <span>|</span>
            <span>TIME: {currentTime.toFixed(1)}s / {maxTime.toFixed(1)}s</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            <span>SYSTEM ONLINE</span>
          </div>
        </div>
      </footer>
      <Footer />
    </div>
  );
};

export default Index;
