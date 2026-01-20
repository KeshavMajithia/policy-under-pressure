import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ArrowLeft, Activity, TrendingDown } from 'lucide-react';
import { Link } from 'react-router-dom';
import { SegmentNav } from '@/components/SegmentNav';
import { Footer } from '@/components/Footer';
import { SegmentHeader } from '@/components/SegmentHeader';

const GradientChart = ({ title, data, xKey, yKey, description, insight, xLabel, yLabel, domain, reversed }: {
    title: string;
    data: any;
    xKey: string;
    yKey: string;
    description: string;
    insight: string;
    xLabel: string;
    yLabel: string;
    domain?: [number | string, number | string];
    reversed?: boolean;
}) => {
    if (!data || !data.RL || !data.ES) return null;

    const chartData = data.RL.map((point, i) => {
        return {
            [xKey]: point.level,
            RL: point[yKey],
            ES: data.ES[i] ? data.ES[i][yKey] : 0,
        };
    });

    return (
        <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted overflow-hidden">
            <div className="p-6">
                <div className="flex justify-between items-start mb-6">
                    <div>
                        <h3 className="text-xl font-display font-bold text-foreground tracking-wide mb-2 flex items-center gap-2">
                            <TrendingDown className="w-5 h-5 text-primary" /> {title}
                        </h3>
                        <p className="text-sm text-muted-foreground max-w-2xl">
                            {description}
                        </p>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Chart (Left - 2cols) */}
                    <div className="lg:col-span-2">
                        <div className="h-[250px] sm:h-[300px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                    <XAxis
                                        dataKey={xKey}
                                        stroke="#888888"
                                        label={{ value: xLabel, position: 'insideBottom', offset: -8 }}
                                        reversed={reversed}
                                        tick={{ fontSize: 10 }}
                                    />
                                    <YAxis
                                        stroke="#888888"
                                        label={{ value: yLabel, angle: -90, position: 'insideLeft' }}
                                        domain={domain || ['auto', 'auto']}
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
                                    <Line
                                        type="monotone"
                                        dataKey="RL"
                                        stroke="#3b82f6"
                                        strokeWidth={2}
                                        dot={{ r: 3, fill: '#3b82f6' }}
                                        activeDot={{ r: 5 }}
                                        name="RL (PPO)"
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="ES"
                                        stroke="#ef4444"
                                        strokeWidth={2}
                                        dot={{ r: 3, fill: '#ef4444' }}
                                        activeDot={{ r: 5 }}
                                        name="ES (Evolution)"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Insight (Right - 1col) */}
                    <div className="space-y-4">
                        <div className="p-4 bg-muted/20 rounded-lg border border-white/5">
                            <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">Key Insight</h4>
                            <p className="text-sm leading-relaxed text-foreground/90">
                                {insight}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const Segment3 = () => {
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        const loadData = async () => {
            const timestamp = Date.now();
            try {
                const res = await fetch(`/gradient_results_v2.json?t=${timestamp}`);
                if (res.ok) {
                    const json = await res.json();
                    setData(json);
                }
            } catch (e) {
                console.warn("Failed to fetch gradient results v2", e);
            }
        };
        loadData();
    }, []);

    if (!data) return <div className="min-h-screen flex items-center justify-center text-muted-foreground">Loading Gradient Data...</div>;

    return (
        <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/20">
            <SegmentHeader
                title="SEGMENT 3:"
                neonText="GRADIENT"
                subtitle="Optimization Landscapes: The Death of RL"
                color="white"
            />

            <main className="container mx-auto px-4 py-8 space-y-12 mt-16">
                <div className="max-w-4xl mx-auto space-y-4 text-center">
                    <h2 className="text-2xl font-bold text-foreground">Quantitative Robustness Analysis</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        This segment provides <strong>statistical analysis</strong> across severity gradients. Each chart aggregates data from 5 seeds per level.
                        For individual race replays at specific severity levels, see <Link to="/segment2" className="text-primary underline">Segment 2</Link>.
                    </p>
                    <p className="text-sm text-muted-foreground">
                        Focus: Measuring exact failure thresholds and degradation slopes using behavioral metrics (Speed, Lateral Error, Steering Variance).
                    </p>
                </div>

                <div className="space-y-8">
                    {/* Friction Ladder - Mean Speed */}
                    <GradientChart
                        title="1. Friction Ladder: Performance"
                        data={data.friction}
                        xKey="level"
                        yKey="mean_speed"
                        xLabel="Friction Coefficient (μ)"
                        yLabel="Mean Speed (m/s)"
                        reversed={true}
                        description="Reducing tire friction from 1.0 (Normal) to 0.2 (Ice). Measuring mean speed as performance indicator."
                        insight="RL: Policy collapse revealed—maintains near-zero speed regardless of friction (already at worst-case before testing). ES: Speed increases = uncontrolled sliding, not adaptation. Both fail to maintain task performance."
                    />

                    {/* Friction Ladder - Lateral Error */}
                    <GradientChart
                        title="1. Friction Ladder: Control Precision"
                        data={data.friction}
                        xKey="level"
                        yKey="lat_error_rms"
                        xLabel="Friction Coefficient (μ)"
                        yLabel="Lateral Error RMS (m)"
                        reversed={true}
                        description="Measuring lateral error RMS as control precision degrades with friction."
                        insight="RL: Stable error because it doesn't move (trivial solution). ES: Increasing error confirms loss of traction control. Neither achieves robust task completion."
                    />

                    {/* Noise Gradient - Mean Speed */}
                    <GradientChart
                        title="2. Noise Gradient: Performance"
                        data={data.noise}
                        xKey="level"
                        yKey="mean_speed"
                        xLabel="Noise Std Dev (σ)"
                        yLabel="Mean Speed (m/s)"
                        description="Injecting Gaussian noise into sensor readings. Measuring performance retention."
                        insight="RL: Noise immunity is an artifact of policy collapse (no movement = no sensor dependency). ES: Speed degradation reveals genuine observation-dependent control. Both fail, but for different reasons."
                    />

                    {/* Noise Gradient - Lateral Error */}
                    <GradientChart
                        title="2. Noise Gradient: Control Precision"
                        data={data.noise}
                        xKey="level"
                        yKey="lat_error_rms"
                        xLabel="Noise Std Dev (σ)"
                        yLabel="Lateral Error RMS (m)"
                        description="Measuring lateral error growth under sensor noise."
                        insight="RL: Minimal error change (already at floor performance). ES: High error regardless of noise level— crashes immediately in all conditions. Task difficulty insufficient to differentiate."
                    />

                    {/* Delay Spectrum */}
                    <GradientChart
                        title="3. Delay Spectrum: Credit Assignment"
                        data={data.delay}
                        xKey="level"
                        yKey="mean_speed"
                        xLabel="Reward Delay (Steps)"
                        yLabel="Mean Speed (m/s)"
                        description="Delaying the reward signal by N steps (0 to 20). Testing credit assignment."
                        insight="RL: Delay irrelevant (policy collapse predates testing). ES: Erratic response suggests episodic evaluation breaks down when reward signal is delayed. Neither demonstrates adaptation."
                    />

                    {/* Action Masking */}
                    <GradientChart
                        title="4. Action Masking: Control Redundancy"
                        data={data.mask}
                        xKey="level"
                        yKey="steering_variance"
                        xLabel="Drop Probability (p)"
                        yLabel="Steering Variance"
                        description="Randomly dropping actions (coasting) with probability p. Testing control style."
                        insight="RL: Low variance because minimal steering needed (near-stasis). ES: High variance = constant corrections; drops with masking (artifact of intervention itself). Both fail to demonstrate control redundancy."
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
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Policy collapse during a-boost (speed →0)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>No benefit from extra gradient steps</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-yellow-400 mt-0.5">~</span>
                                    <span>Stuck in local minimum (never recovers)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Low steering variance (minimal control)</span>
                                </li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-semibold text-red-400 text-sm uppercase tracking-wide mb-3">ES Characteristics</h3>
                            <ul className="space-y-2 text-sm text-foreground/80">
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Survives a-boost (slower but functional)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Benefits from additional evolution cycles</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Continuous improvement trajectory</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>High steering variance (active control)</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-6 italic leading-relaxed">
                        Alpha-boost experiment reveals critical difference: RL's gradient-based learning trapped in poor basin,
                        while ES's population-based search continues exploring. This isn't just sample efficiency—it's optimization landscape topology.
                    </p>
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default Segment3;
