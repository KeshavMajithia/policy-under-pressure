
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Play, Pause, RotateCcw, Activity, Zap, Shield } from 'lucide-react';
import { Link } from 'react-router-dom';
import { SimpleRaceCanvas } from '@/components/SimpleRaceCanvas';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { SegmentNav } from '@/components/SegmentNav';
import { Footer } from '@/components/Footer';
import { SegmentHeader } from '@/components/SegmentHeader';

const ExperimentCard = ({ title, experiment, data, description, insight, icon: Icon }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [progress, setProgress] = useState(0);
    const [selectedLevel, setSelectedLevel] = useState(null);

    // Extract levels from data
    const levels = data ? Object.keys(data) : [];
    const currentLevel = selectedLevel || levels[0];
    const currentData = data?.[currentLevel];

    // Get trajectories
    const rlTraj = currentData?.RL?.sample_trajectory || [];
    const esTraj = currentData?.ES?.sample_trajectory || [];
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
                    return p + 0.01;
                });
            }, 50);
        }
        return () => clearInterval(interval);
    }, [isPlaying]);

    const handleReset = () => {
        setIsPlaying(false);
        setProgress(0);
    };

    const idx = Math.floor(progress * maxSteps);
    const rlState = rlTraj[Math.min(idx, rlTraj.length - 1)] || null;
    const esState = esTraj[Math.min(idx, esTraj.length - 1)] || null;

    // Compute bounds
    const allPoints = [...rlTraj, ...esTraj];
    const xs = allPoints.map(p => p.x);
    const ys = allPoints.map(p => p.y);
    const minX = Math.min(...xs, 0);
    const maxX = Math.max(...xs, 100);
    const minY = Math.min(...ys, 0);
    const maxY = Math.max(...ys, 60);

    const bounds = [
        [minX - 5, minY - 5],
        [maxX + 5, minY - 5],
        [maxX + 5, maxY + 5],
        [minX - 5, maxY + 5],
        [minX - 5, minY - 5]
    ] as [number, number][];

    // Prepare chart data (behavioral metrics across levels)
    const chartData = levels.map(level => ({
        level: parseFloat(level),
        RL_Speed: data[level]?.RL?.aggregated?.mean_speed || 0,
        ES_Speed: data[level]?.ES?.aggregated?.mean_speed || 0,
        RL_LatError: data[level]?.RL?.aggregated?.lat_error_rms || 0,
        ES_LatError: data[level]?.ES?.aggregated?.lat_error_rms || 0,
        RL_Survival: data[level]?.RL?.aggregated?.survival_rate || 0,
        ES_Survival: data[level]?.ES?.aggregated?.survival_rate || 0
    }));

    return (
        <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted overflow-hidden">
            <div className="p-6">
                {/* Header */}
                <div className="flex justify-between items-start mb-6">
                    <div>
                        <h3 className="text-xl font-display font-bold text-foreground tracking-wide mb-2 flex items-center gap-2">
                            {Icon && <Icon className="w-5 h-5 text-primary" />} {title}
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
                        {/* Level Selector */}
                        <div className="flex gap-2 flex-wrap">
                            {levels.map(level => (
                                <button
                                    key={level}
                                    onClick={() => setSelectedLevel(level)}
                                    className={`px-3 py-1 rounded text-sm font-mono transition-colors ${level === currentLevel
                                        ? 'bg-primary text-primary-foreground'
                                        : 'bg-muted/20 text-muted-foreground hover:bg-muted/40'
                                        }`}
                                >
                                    {experiment === 'exp_1' ? `μ=${level}` : `σ=${level}`}
                                </button>
                            ))}
                        </div>

                        {/* Canvas */}
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
                                <button onClick={handleReset} className="p-2 bg-secondary/20 hover:bg-secondary/40 rounded-full text-secondary transition-colors">
                                    <RotateCcw className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        {/* Behavioral Metrics Chart */}
                        <div className="h-[250px] sm:h-[300px] bg-black/20 rounded-lg p-4 border border-white/5">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                    <XAxis
                                        dataKey="level"
                                        stroke="#888888"
                                        label={{ value: experiment === 'exp_1' ? 'Friction' : 'Noise', position: 'insideBottom', offset: -8, style: { fontSize: 10 } }}
                                        reversed={experiment === 'exp_1'}
                                        tick={{ fontSize: 10 }}
                                    />
                                    <YAxis
                                        stroke="#888888"
                                        label={{ value: 'Speed (m/s)', angle: -90, position: 'insideLeft', style: { fontSize: 10 } }}
                                        tick={{ fontSize: 10 }}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333', borderRadius: '8px', fontSize: '11px' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <Legend
                                        layout="horizontal"
                                        verticalAlign="top"
                                        align="center"
                                        wrapperStyle={{ paddingBottom: '10px', fontSize: '11px' }}
                                    />
                                    <Line type="monotone" dataKey="RL_Speed" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="RL Speed" />
                                    <Line type="monotone" dataKey="ES_Speed" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="ES Speed" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Metrics / Insight (Right - 1col) */}
                    <div className="space-y-6">
                        <div className="p-4 bg-muted/20 rounded-lg border border-white/5">
                            <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">Key Insight</h4>
                            <p className="text-sm leading-relaxed text-foreground/90">
                                {insight}
                            </p>
                        </div>

                        {/* Current Level Metrics */}
                        {currentData && (
                            <div className="space-y-3">
                                <h4 className="text-xs uppercase tracking-wider text-muted-foreground">Current Level: {currentLevel}</h4>
                                <div className="grid grid-cols-2 gap-2 text-xs">
                                    <div className="p-2 bg-rl/10 border border-rl/20 rounded">
                                        <div className="text-muted-foreground">RL Speed</div>
                                        <div className="text-rl font-mono font-bold">{currentData.RL?.aggregated?.mean_speed?.toFixed(3)} m/s</div>
                                    </div>
                                    <div className="p-2 bg-es/10 border border-es/20 rounded">
                                        <div className="text-muted-foreground">ES Speed</div>
                                        <div className="text-es font-mono font-bold">{currentData.ES?.aggregated?.mean_speed?.toFixed(3)} m/s</div>
                                    </div>
                                    <div className="p-2 bg-rl/10 border border-rl/20 rounded">
                                        <div className="text-muted-foreground">RL Lat Error</div>
                                        <div className="text-rl font-mono font-bold">{currentData.RL?.aggregated?.lat_error_rms?.toFixed(2)} m</div>
                                    </div>
                                    <div className="p-2 bg-es/10 border border-es/20 rounded">
                                        <div className="text-muted-foreground">ES Lat Error</div>
                                        <div className="text-es font-mono font-bold">{currentData.ES?.aggregated?.lat_error_rms?.toFixed(2)} m</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

const Segment2 = () => {
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        const loadData = async () => {
            const timestamp = Date.now();
            try {
                const res = await fetch(`/segment2_results_v2.json?t=${timestamp}`);
                if (res.ok) {
                    const json = await res.json();
                    setData(json);
                }
            } catch (e) {
                console.warn("Failed to fetch segment2 results v2", e);
            }
        };
        loadData();
    }, []);

    if (!data) return <div className="min-h-screen flex items-center justify-center text-muted-foreground">Loading Experiment Data...</div>;

    return (
        <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/20">
            <SegmentHeader
                title="SEGMENT 2:"
                neonText="ROBUSTNESS"
                subtitle="Adversarial Physics: Testing Agent Resilience"
                color="secondary"
            />

            <main className="container mx-auto px-4 py-8 space-y-12 mt-16">
                <div className="max-w-4xl mx-auto space-y-4 text-center">
                    <h2 className="text-2xl font-bold text-foreground">Progressive Degradation</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        We apply increasing stress via <strong>Friction Ladders</strong> and <strong>Noise Gradients</strong>.
                        Using behavioral metrics (Speed, Lateral Error), we measure how gracefully each agent degrades.
                    </p>
                </div>

                <div className="space-y-8">
                    {/* Exp 1: Friction Ladder */}
                    <ExperimentCard
                        title="1. The Friction Ladder (Physics Robustness)"
                        experiment="exp_1"
                        icon={Shield}
                        data={data.exp_1}
                        description="Progressive friction reduction from 1.0 (Normal) to 0.2 (Ice). Testing control under degraded physics."
                        insight="RL exhibits policy collapse: it 'survives' by avoiding the task entirely (~0.01 m/s). This is degenerate behavior, not robustness. ES maintains task-directed motion (high speed) but loses control, revealing uncontrolled sliding rather than adaptation."
                    />

                    {/* Exp 2: Noise Gradient */}
                    <ExperimentCard
                        title="2. The Noise Gradient (Sensor Robustness)"
                        experiment="exp_2"
                        icon={Zap}
                        data={data.exp_2}
                        description="Gaussian noise injection (σ = 0.0 to 0.3) into sensor readings. Testing sensitivity to observation corruption."
                        insight="RL is unaffected by noise—not due to robustness, but because its degenerate policy doesn't rely on observations (near-zero speed = no control needed). ES degrades as noise increases, showing genuine dependence on sensor quality for speed control."
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
                                    <span>Optimal friction performance (μ=1.0)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Policy collapse at μ&lt;0.6 (near-zero speed)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-yellow-400 mt-0.5">~</span>
                                    <span>Noise insensitive (degenerate policy doesn't use observations)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Catastrophic velocity masking failure</span>
                                </li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-semibold text-red-400 text-sm uppercase tracking-wide mb-3">ES Characteristics</h3>
                            <ul className="space-y-2 text-sm text-foreground/80">
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Graceful friction degradation curve</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Better low-friction robustness (μ=0.3)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Degrades under noise (relies on sensors)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Catastrophic velocity masking failure</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-6 italic leading-relaxed">
                        Quantitative analysis reveals RL's degenerate policy—numeric optimality masks behavioral collapse.
                        ES's graceful degradation suggests more robust policy structure, but both architectures fundamentally depend on full state observability.
                    </p>
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default Segment2;
