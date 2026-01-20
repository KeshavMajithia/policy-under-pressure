
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play, Pause, RotateCcw, Activity } from 'lucide-react';
import { Link } from 'react-router-dom';
import { SimpleRaceCanvas } from '@/components/SimpleRaceCanvas';
import { SegmentNav } from '@/components/SegmentNav';
import { Footer } from '@/components/Footer';
import { SegmentHeader } from '@/components/SegmentHeader';

// Constants for canvas scaling - matching standard view
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 450;

const ExperimentCard = ({ title, experiment, data, description, insight }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0); // 0 to 1

    // Derived from data
    const rlTraj = data?.RL?.[experiment]?.sample_trajectory || [];
    const esTraj = data?.ES?.[experiment]?.sample_trajectory || [];
    const maxSteps = Math.max(rlTraj.length, esTraj.length, 1);

    useEffect(() => {
        let interval;
        if (isPlaying) {
            interval = setInterval(() => {
                setProgress(p => {
                    if (p >= 1) {
                        setIsPlaying(false);
                        return 1;
                    }
                    return p + 0.01; // ~100 steps to finish
                });
            }, 50);
        }
        return () => clearInterval(interval);
    }, [isPlaying]);

    const handleReset = () => {
        setIsPlaying(false);
        setProgress(0);
    };

    // Get current state based on progress
    const idx = Math.floor(progress * maxSteps);
    const rlState = rlTraj[Math.min(idx, rlTraj.length - 1)] || null;
    const esState = esTraj[Math.min(idx, esTraj.length - 1)] || null;

    // Metrics Extraction
    const rlMetrics = data?.RL?.[experiment]?.metrics || {};
    const esMetrics = data?.ES?.[experiment]?.metrics || {};

    // Auto-scale: Create a bounding box 'track' to force RaceCanvas to zoom correctly
    const allPoints = [...rlTraj, ...esTraj];
    const xs = allPoints.map(p => p.x);
    const ys = allPoints.map(p => p.y);
    const minX = Math.min(...xs, 0);
    const maxX = Math.max(...xs, 100);
    const minY = Math.min(...ys, 0);
    const maxY = Math.max(...ys, 60);

    // Invisible boundary box (RaceCanvas will draw this, but it frames the action)
    const bounds = [
        [minX - 5, minY - 5],
        [maxX + 5, minY - 5],
        [maxX + 5, maxY + 5],
        [minX - 5, maxY + 5],
        [minX - 5, minY - 5]
    ] as [number, number][];

    return (
        <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted overflow-hidden">
            <div className="p-6">
                {/* Header */}
                <div className="flex justify-between items-start mb-6">
                    <div>
                        <h3 className="text-xl font-display font-bold text-foreground tracking-wide mb-2">
                            {title}
                        </h3>
                        <p className="text-sm text-muted-foreground max-w-2xl">
                            {description}
                        </p>
                    </div>
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                    {/* Visualizer (Left - 2cols) */}
                    <div className="lg:col-span-2 space-y-4">
                        <div className="relative rounded-lg border border-track/20 bg-black/40 h-[300px] overflow-hidden flex items-center justify-center">
                            <SimpleRaceCanvas
                                trackPoints={bounds}
                                rlState={rlState}
                                esState={esState}
                                rlTrail={rlTraj.slice(0, idx)}
                                esTrail={esTraj.slice(0, idx)}
                            />

                            {/* Overlay Controls */}
                            <div className="absolute bottom-4 left-4 flex gap-2">
                                <button onClick={() => setIsPlaying(!isPlaying)} className="p-2 bg-primary/20 hover:bg-primary/40 rounded-full text-primary transition-colors">
                                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                                </button>
                                <button onClick={handleReset} className="p-2 bg-muted/20 hover:bg-muted/40 rounded-full text-muted-foreground transition-colors">
                                    <RotateCcw className="w-4 h-4" />
                                </button>
                            </div>

                            {/* Labels */}
                            <div className="absolute top-4 right-4 flex flex-col gap-2 font-mono text-[10px]">
                                <div className="text-rl flex items-center gap-1">
                                    <div className="w-2 h-2 rounded-full bg-rl" /> RL (PPO)
                                </div>
                                <div className="text-es flex items-center gap-1">
                                    <div className="w-2 h-2 rounded-full bg-es" /> ES (Evo)
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Metrics & Insight (Right - 1col) */}
                    <div className="space-y-6">
                        {/* Metrics Table */}
                        <div className="bg-background/40 rounded-lg p-4 border border-muted">
                            <h4 className="text-xs font-mono text-muted-foreground uppercase mb-3">Key Metrics</h4>
                            <div className="space-y-3 font-mono text-sm">
                                {Object.keys(rlMetrics).map(key => (
                                    <div key={key} className="flex justify-between items-center pb-2 border-b border-muted/50 last:border-0">
                                        <span className="text-muted-foreground capitalize">{key.replace('_', ' ')}</span>
                                        <div className="flex gap-4">
                                            <span className="text-rl">{typeof rlMetrics[key] === 'number' ? rlMetrics[key].toFixed(2) : rlMetrics[key]}</span>
                                            <span className="text-muted-foreground">vs</span>
                                            <span className="text-es">{typeof esMetrics[key] === 'number' ? esMetrics[key].toFixed(2) : esMetrics[key]}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Insight Block */}
                        <div className="bg-primary/5 rounded-lg p-4 border border-primary/20">
                            <div className="flex items-center gap-2 mb-2 text-primary">
                                <Activity className="w-4 h-4" />
                                <span className="text-xs font-bold uppercase tracking-wider">Research Verdict</span>
                            </div>
                            <p className="text-sm text-foreground/80 leading-relaxed">
                                {insight}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const Segment1 = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/experiment_results.json')
            .then(res => res.json())
            .then(d => {
                setData(d);
                setLoading(false);
            })
            .catch(err => {
                console.error("Failed to load experiments", err);
                setLoading(false);
            });
    }, []);

    if (loading) return <div className="min-h-screen bg-background grid-bg flex items-center justify-center text-primary font-mono">LOADING EXPERIMENT DATA...</div>;

    return (
        <div className="min-h-screen bg-background grid-bg pb-20">
            <SegmentHeader
                title="SEGMENT 1:"
                neonText="FRAGILITY"
                subtitle="Behavioral Stress Tests: RL vs ES"
                color="primary"
            />

            {/* Content */}
            <main className="container mx-auto px-4 py-8 space-y-12">

                <div className="max-w-4xl mx-auto space-y-4 text-center">
                    <h2 className="text-2xl font-bold text-foreground">The Fragility of Excellence</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        In Phase 2, we froze the "Perfect" agents from Phase 1 and subjected them to 5 brutal stress tests.
                        The results confirm that <strong>RL optimizes for Peak Performance</strong> (fragile), while <strong>ES optimizes for Survival</strong> (robust).
                    </p>
                </div>

                <div className="space-y-8">
                    {/* Exp 1: New World */}
                    <ExperimentCard
                        title="1. The New World (Generalization)"
                        experiment="exp_1"
                        data={data}
                        description="HEAD-TO-HEAD: Both agents are dropped onto the EXACT SAME complex procedural track (Shared Seed). This FAIRLY tests if they can drive on unknown terrain or if they just memorized the training loop."
                        insight="Observe the traces. If RL (Blue) fails on the same curve where ES (Red) survives, it suggests ES has a more robust generalized policy. If both fail, the track complexity exceeds their architectural capacity."
                    />

                    {/* Exp 2: Ice Patch */}
                    <ExperimentCard
                        title="2. The Ice Patch (Physics Robustness)"
                        experiment="exp_2"
                        data={data}
                        description="Mid-race, we suddenly reduce friction by 70% for 2 seconds. This simulates hitting a patch of ice."
                        insight="STRONG EMPIRICAL EVIDENCE. RL (driving at the limit) flies off the track. ES (driving with safety margins) slides but deviates significantly less, demonstrating higher robustness to physics shifts."
                    />

                    {/* Exp 3: Foggy Sensor */}
                    <ExperimentCard
                        title="3. The Foggy Sensor (Noise)"
                        experiment="exp_3"
                        data={data}
                        description="We inject Gaussian noise into the Hearing and Lateral Error sensors to simulate poor perception."
                        insight="Both agents struggle. ES's wobbles are dampened by its smoother control law, while RL's stiff policy becomes jittery. However, the high noise level caused failure for both, suggesting a need for 'graceful degradation' tests."
                    />

                    {/* Exp 4: Blindfold */}
                    <ExperimentCard
                        title="4. The Blindfold (Partial Observability)"
                        experiment="exp_4"
                        data={data}
                        description="We cut the Velocity sensor wire. The agent must drive 'blind' to its own speed."
                        insight="Total failure for both. Hard-zeroing velocity is an extreme event that breaks the agents' reliance on the full state tuple (x,y,h,v). Future tests should use intermittent masking."
                    />

                    {/* Exp 5: Wake Up Call */}
                    <ExperimentCard
                        title="5. The Wake Up Call (Adversarial Starts)"
                        experiment="exp_5"
                        data={data}
                        description="Agents are spawned facing a wall or in the middle of a sharp turn."
                        insight="Shared Failure. Both agents learned 'forward-only' policies and lack the 'reverse gear' logic needed to recover from walls. This is a design architecture limitation, not just a learning failure."
                    />
                </div>

                {/* Research Synthesis */}
                <div className="bg-gradient-to-br from-purple-500/10 via-blue-500/10 to-background border border-purple-500/20 rounded-xl p-8">
                    <h2 className="text-2xl font-bold text-foreground mb-6">Research Synthesis</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div>
                            <h3 className="font-semibold text-blue-400 text-sm uppercase tracking-wide mb-3">RL Characteristics</h3>
                            <ul className="space-y-2 text-sm text-foreground/80">
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Perfect figure-8 navigation (0.1m lateral error)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Fails on novel terrain (generalization)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Breaks under ice patch (physics shift)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>No recovery from adversarial starts</span>
                                </li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-semibold text-red-400 text-sm uppercase tracking-wide mb-3">ES Characteristics</h3>
                            <ul className="space-y-2 text-sm text-foreground/80">
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Smoother control (lower jerk)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Better ice patch survival (slides less)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Higher lateral error (less precise)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>No recovery from adversarial starts</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-6 italic leading-relaxed">
                        RL optimizes for peak performance (fragile), while ES optimizes for survival (robust).
                        Both architectures lack reverse recovery logic—a fundamental design limitation requiring architectural revision.
                    </p>
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default Segment1;
