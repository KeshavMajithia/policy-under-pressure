import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Network } from 'lucide-react';

interface ResearchChartsProps {
  data: any; // Raw JSON from experiment
}

export const ResearchCharts = ({ data }: ResearchChartsProps) => {
  if (!data) return null;
  
  const chartData = useMemo(() => {
     // Transform data for Recharts
     // We want format: [{noise: 0.1, rl: 30, es: 45}, ...]
     
     const noiseLevels = data.metadata.noise_levels;
     const rlResults = data.results.RL;
     const esResults = data.results.ES;
     
     return noiseLevels.map((noise: number, idx: number) => {
         // Find matching result for this noise level (assumed ordered)
         const rl = rlResults.find((r: any) => r.noise === noise);
         const es = esResults.find((r: any) => r.noise === noise);
         
         return {
             noise: noise,
             rlMean: rl?.mean_reward ?? 0,
             esMean: es?.mean_reward ?? 0,
         };
     });
  }, [data]);

  return (
    <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted p-4 mt-6">
      <div className="flex items-center gap-2 mb-4">
        <Network className="w-4 h-4 text-primary" />
        <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
          Research Finding: {data.metadata.experiment}
        </h3>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis
              dataKey="noise"
              label={{ value: 'Noise Level (Std Dev)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }}
              stroke="#64748b"
              tick={{ fontSize: 10 }}
            />
            <YAxis
              label={{ value: 'Mean Reward', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }}
              stroke="#64748b"
              tick={{ fontSize: 10 }}
            />
            <Tooltip
                contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid #333' }}
                itemStyle={{ fontSize: '12px' }}
                labelStyle={{ fontSize: '12px', color: '#888' }}
            />
            <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
            
            <Line
              type="monotone"
              dataKey="rlMean"
              name="RL (PPO)"
              stroke="#00d4ff"
              strokeWidth={3}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="esMean"
              name="ES (OpenAI)"
              stroke="#ff3366"
              strokeWidth={3}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 p-3 bg-background/40 rounded-lg border border-white/5">
        <h4 className="text-sm font-bold text-foreground mb-2">Analysis</h4>
        <p className="text-xs text-muted-foreground leading-relaxed">
          The <strong>Evolution Strategies (ES)</strong> agent demonstrates superior robustness to sensor noise compared to PPO. 
          While PPO performance degrades linearly as noise increases (Policy Collapse), the ES population-based smoothing 
          creates a flatter "Wide Valley" of stability, retaining competence even at high noise levels (1.0).
        </p>
      </div>
    </div>
  );
};
