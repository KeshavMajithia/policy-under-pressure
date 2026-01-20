import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, TrendingUp, Shield, Cpu, ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';
import { SegmentNav } from '@/components/SegmentNav';
import { Footer } from '@/components/Footer';
import { SegmentHeader } from '@/components/SegmentHeader';

const Segment4 = () => {
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        Promise.all([
            fetch('/segment4_exploit.json').then(r => r.json()),
            fetch('/segment4_sensitivity.json').then(r => r.json()),
            fetch('/segment4_alignment.json').then(r => r.json()),
            fetch('/segment4_scaling.json').then(r => r.json())
        ]).then(([exploit, sensitivity, alignment, scaling]) => {
            setData({ exploit, sensitivity, alignment, scaling });
        }).catch(err => console.error('Error loading data:', err));
    }, []);

    if (!data) {
        return (
            <div className="min-h-screen bg-background grid-bg flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            </div>
        );
    }

    // Prepare chart data
    const exploitChartData = [
        { name: 'RL Exploit', value: data.exploit.aggregated.RL_exploit.exploit_ratio * 100, fill: '#ef4444' },
        { name: 'RL Baseline', value: data.exploit.aggregated.RL_baseline.exploit_ratio * 100, fill: '#6b7280' },
        { name: 'ES Exploit', value: data.exploit.aggregated.ES_exploit.exploit_ratio * 100, fill: '#f97316' },
        { name: 'ES Baseline', value: data.exploit.aggregated.ES_baseline.exploit_ratio * 100, fill: '#9ca3af' }
    ];

    const sensitivityLineData = ['minus_20', 'minus_10', 'baseline', 'plus_10', 'plus_20'].map(config => ({
        config,
        RL: data.sensitivity.aggregated.RL[config]?.mean_divergence || 0,
        ES: data.sensitivity.aggregated.ES[config]?.mean_divergence || 0
    }));

    const alignmentLineData = [
        {
            metric: 'Smoothness',
            'RL Misaligned': data.alignment.aggregated.RL_misaligned.smoothness,
            'ES Misaligned': data.alignment.aggregated.ES_misaligned.smoothness,
            'RL Baseline': data.alignment.aggregated.RL_baseline.smoothness
        },
        {
            metric: 'Jerk',
            'RL Misaligned': data.alignment.aggregated.RL_misaligned.jerk,
            'ES Misaligned': data.alignment.aggregated.ES_misaligned.jerk,
            'RL Baseline': data.alignment.aggregated.RL_baseline.jerk
        },
        {
            metric: 'Entropy',
            'RL Misaligned': data.alignment.aggregated.RL_misaligned.entropy,
            'ES Misaligned': data.alignment.aggregated.ES_misaligned.entropy,
            'RL Baseline': data.alignment.aggregated.RL_baseline.entropy
        }
    ];

    const scalingData = data.scaling.worker_counts.map((workers: number, idx: number) => ({
        workers,
        RL: data.scaling.results.RL[idx].efficiency * 100,
        ES: data.scaling.results.ES[idx].efficiency * 100
    }));

    return (
        <div className="min-h-screen bg-background grid-bg pb-20">
            <SegmentHeader
                title="SEGMENT 4:"
                neonText="REWARD"
                subtitle="Controlled Experiments: Optimizer Behavior Under Reward Manipulation"
                color="amber"
            />

            {/* Content */}
            <main className="container mx-auto px-4 py-8 space-y-12">
                {/* Warning Banner */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-6"
                >
                    <div className="flex items-start gap-4">
                        <AlertTriangle className="w-6 h-6 text-amber-500 mt-0.5 flex-shrink-0" />
                        <div>
                            <h3 className="font-semibold text-amber-500 mb-2">Research Protocol Notice</h3>
                            <p className="text-sm text-foreground/70 leading-relaxed">
                                These agents were trained with <strong>modified objectives</strong> for controlled reward engineering studies.
                                Results characterize optimizer behavior under manipulation, not deployment readiness.
                            </p>
                        </div>
                    </div>
                </motion.div>

                {/* Introduction */}
                <div className="max-w-4xl mx-auto space-y-4 text-center">
                    <h2 className="text-2xl font-bold text-foreground">The Science of Misalignment</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        We deliberately introduce reward exploits, shift coefficients, and misalign objectives to study
                        <strong> how RL vs ES respond to imperfect reward signals</strong>. Unlike Segments 1-3 (which tested frozen agents),
                        here we train NEW agents under modified conditions to understand optimizer-level differences.
                    </p>
                </div>

                {/* Experiment 1: Exploitation */}
                <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted overflow-hidden">
                    <div className="p-6">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h3 className="text-xl font-display font-bold text-foreground tracking-wide mb-2 flex items-center gap-2">
                                    <TrendingUp className="w-5 h-5 text-red-500" /> 1. Reward Exploitation
                                </h3>
                                <p className="text-sm text-muted-foreground max-w-2xl">
                                    We introduce specific loopholes (steering saturation, boundary grazing, wiggle bonus) into the reward function. Who discovers them faster?
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="lg:col-span-2">
                                <div className="h-[250px] sm:h-[300px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={exploitChartData} margin={{ top: 5, right: 5, left: -10, bottom: 5 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                            <XAxis dataKey="name" stroke="#888888" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                                            <YAxis stroke="#888888" label={{ value: 'Exploit Ratio (%)', angle: -90, position: 'insideLeft', style: { fontSize: 10 } }} tick={{ fontSize: 10 }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333', borderRadius: '8px', fontSize: '11px' }}
                                                itemStyle={{ color: '#fff' }}
                                                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                            />
                                            <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                                                {exploitChartData.map((entry, index) => (
                                                    <Cell key={index} fill={entry.fill} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div className="p-4 bg-muted/20 rounded-lg border border-white/5">
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">Key Insight</h4>
                                    <p className="text-sm leading-relaxed text-foreground/90">
                                        RL exploits loopholes <strong className="text-red-400">10x faster</strong> (+867%) while ES shows minimal increase (+11%).
                                        Power vs prudence trade-off.
                                    </p>
                                    <div className="mt-4 space-y-2 text-xs">
                                        <div className="flex justify-between border-b border-border/50 pb-1">
                                            <span className="text-muted-foreground">RL Increase</span>
                                            <span className="font-mono text-red-400">+867%</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">ES Increase</span>
                                            <span className="font-mono text-orange-400">+11%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Experiment 2: Sensitivity */}
                <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted overflow-hidden">
                    <div className="p-6">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h3 className="text-xl font-display font-bold text-foreground tracking-wide mb-2 flex items-center gap-2">
                                    <Shield className="w-5 h-5 text-blue-500" /> 2. Reward Sensitivity
                                </h3>
                                <p className="text-sm text-muted-foreground max-w-2xl">
                                    We vary penalty coefficients (±10%, ±20%) from baseline. How fragile are the learned policies?
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="lg:col-span-2">
                                <div className="h-[250px] sm:h-[300px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={sensitivityLineData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                            <XAxis
                                                dataKey="config"
                                                stroke="#888888"
                                                label={{ value: 'Config', position: 'insideBottom', offset: -8, style: { fontSize: 10 } }}
                                                tick={{ fontSize: 10 }}
                                            />
                                            <YAxis
                                                stroke="#888888"
                                                label={{ value: 'Divergence', angle: -90, position: 'insideLeft', style: { fontSize: 10 } }}
                                                tick={{ fontSize: 10 }}
                                            />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333', borderRadius: '8px', fontSize: '11px' }}
                                                itemStyle={{ color: '#fff' }}
                                                cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }}
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

                            <div className="space-y-4">
                                <div className="p-4 bg-muted/20 rounded-lg border border-white/5">
                                    <div className="mb-3 px-2 py-1 bg-blue-500/10 border border-blue-500/20 rounded inline-block">
                                        <p className="text-xs font-semibold text-blue-400 uppercase tracking-wide">Unexpected</p>
                                    </div>
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">Key Insight</h4>
                                    <p className="text-sm leading-relaxed text-foreground/90">
                                        RL policies remain <strong className="text-green-400">nearly identical</strong> (~0.0005 divergence) across reward changes.
                                        ES generates <strong className="text-red-400">completely different policies</strong> (~1.0 divergence).
                                        <strong className="text-primary"> RL 3500x more stable</strong>.
                                    </p>
                                    <p className="text-xs text-muted-foreground mt-3 italic">
                                        Challenges assumptions about gradient-based brittleness.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Experiment 3: Alignment */}
                <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted overflow-hidden">
                    <div className="p-6">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h3 className="text-xl font-display font-bold text-foreground tracking-wide mb-2 flex items-center gap-2">
                                    <Shield className="w-5 h-5 text-purple-500" /> 3. Alignment Test
                                </h3>
                                <p className="text-sm text-muted-foreground max-w-2xl">
                                    We train agents with a misaligned reward (heavily favors speed over control). Do they learn "correct" driving or literal optimization?
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="lg:col-span-2">
                                <div className="h-[250px] sm:h-[300px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={alignmentLineData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                            <XAxis
                                                dataKey="metric"
                                                stroke="#888888"
                                                tick={{ fontSize: 10 }}
                                            />
                                            <YAxis
                                                stroke="#888888"
                                                label={{ value: 'Value', angle: -90, position: 'insideLeft', style: { fontSize: 10 } }}
                                                tick={{ fontSize: 10 }}
                                            />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333', borderRadius: '8px', fontSize: '11px' }}
                                                itemStyle={{ color: '#fff' }}
                                                cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }}
                                            />
                                            <Legend
                                                layout="horizontal"
                                                verticalAlign="top"
                                                align="center"
                                                wrapperStyle={{ paddingBottom: '10px', fontSize: '11px' }}
                                            />
                                            <Line type="monotone" dataKey="RL Misaligned" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                                            <Line type="monotone" dataKey="ES Misaligned" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                                            <Line type="monotone" dataKey="RL Baseline" stroke="#10b981" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 3 }} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div className="p-4 bg-muted/20 rounded-lg border border-white/5">
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">Key Insight</h4>
                                    <p className="text-sm leading-relaxed text-foreground/90">
                                        RL maintains driving quality despite misalignment (smoothness 8.52 vs 8.43).
                                        ES degrades significantly (0.95 vs 1.62).
                                        RL learns robust representations that transcend specific reward signals.
                                    </p>
                                    <div className="mt-4 space-y-2 text-xs">
                                        <div>
                                            <div className="flex justify-between mb-1">
                                                <span className="text-muted-foreground">RL Change</span>
                                                <span className="text-green-400">+1%</span>
                                            </div>
                                            <div className="h-1.5 bg-muted/50 rounded-full overflow-hidden">
                                                <div className="h-full bg-green-500 rounded-full" style={{ width: '99%' }}></div>
                                            </div>
                                        </div>
                                        <div>
                                            <div className="flex justify-between mb-1">
                                                <span className="text-muted-foreground">ES Change</span>
                                                <span className="text-red-400">-41%</span>
                                            </div>
                                            <div className="h-1.5 bg-muted/50 rounded-full overflow-hidden">
                                                <div className="h-full bg-red-500 rounded-full" style={{ width: '59%' }}></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Experiment 4: Scaling */}
                <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted overflow-hidden">
                    <div className="p-6">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h3 className="text-xl font-display font-bold text-foreground tracking-wide mb-2 flex items-center gap-2">
                                    <Cpu className="w-5 h-5 text-green-500" /> 4. Compute Scaling
                                </h3>
                                <p className="text-sm text-muted-foreground max-w-2xl">
                                    We benchmark training speed across different worker counts. Who benefits from parallelization?
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="lg:col-span-2">
                                <div className="h-[250px] sm:h-[300px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={scalingData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                            <XAxis
                                                dataKey="workers"
                                                stroke="#888888"
                                                label={{ value: 'Workers', position: 'insideBottom', offset: -8, style: { fontSize: 10 } }}
                                                tick={{ fontSize: 10 }}
                                            />
                                            <YAxis
                                                stroke="#888888"
                                                label={{ value: 'Efficiency (%)', angle: -90, position: 'insideLeft', style: { fontSize: 10 } }}
                                                domain={[0, 110]}
                                                tick={{ fontSize: 10 }}
                                            />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#333', borderRadius: '8px', fontSize: '11px' }}
                                                itemStyle={{ color: '#fff' }}
                                                cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }}
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

                            <div className="space-y-4">
                                <div className="p-4 bg-muted/20 rounded-lg border border-white/5">
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">Key Insight</h4>
                                    <p className="text-sm leading-relaxed text-foreground/90">
                                        Both show poor scaling beyond 2 workers. ES peaks at 103% with 2 workers then drops.
                                        Simulated parallelism—real distributed training may differ.
                                    </p>
                                    <div className="mt-4 space-y-1 text-xs">
                                        <div className="flex justify-between border-b border-border/50 pb-1">
                                            <span className="text-muted-foreground">RL @ 8 workers</span>
                                            <span className="font-mono text-blue-400">20.9%</span>
                                        </div>
                                        <div className="flex justify-between border-b border-border/50 pb-1">
                                            <span className="text-muted-foreground">ES @ 8 workers</span>
                                            <span className="font-mono text-red-400">13.2%</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">ES @ 2 workers</span>
                                            <span className="font-mono text-green-400">102.8%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Summary */}
                <div className="bg-gradient-to-br from-purple-500/10 via-blue-500/10 to-background border border-purple-500/20 rounded-xl p-8">
                    <h2 className="text-2xl font-bold text-foreground mb-6">Research Synthesis</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div>
                            <h3 className="font-semibold text-blue-400 text-sm uppercase tracking-wide mb-3">RL Characteristics</h3>
                            <ul className="space-y-2 text-sm text-foreground/80">
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Discovers exploits aggressively (+867%)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Highly stable to reward changes (3500x)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Maintains alignment under misalignment</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Poor parallel scaling (21% @ 8 workers)</span>
                                </li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-semibold text-red-400 text-sm uppercase tracking-wide mb-3">ES Characteristics</h3>
                            <ul className="space-y-2 text-sm text-foreground/80">
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Resists exploitation (+11% only)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Highly sensitive to reward changes</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-red-400 mt-0.5">✗</span>
                                    <span>Degrades under misalignment (-41%)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="text-green-400 mt-0.5">✓</span>
                                    <span>Good 2-worker scaling (103%)</span>
                                </li>
                            </ul>
                        </div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-6 italic leading-relaxed">
                        These controlled experiments reveal nuanced trade-offs. Neither algorithm dominates across all dimensions.
                        RL shows surprising robustness to reward perturbations, challenging conventional wisdom about gradient-based fragility.
                    </p>
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default Segment4;
