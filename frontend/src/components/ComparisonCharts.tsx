import { useState, useMemo } from 'react';
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
import { Switch } from '@/components/ui/switch';
import { RaceData } from '@/types/race';

interface ComparisonChartsProps {
  rlData: RaceData | null;
  esData: RaceData | null;
  currentTime: number;
}

export const ComparisonCharts = ({
  rlData,
  esData,
  currentTime,
}: ComparisonChartsProps) => {
  const [showCumulative, setShowCumulative] = useState(true);

  const chartData = useMemo(() => {
    if (!rlData && !esData) return [];

    const maxLen = Math.max(
      rlData?.trajectory.length ?? 0,
      esData?.trajectory.length ?? 0
    );

    const data = [];
    let rlCum = 0;
    let esCum = 0;

    for (let i = 0; i < maxLen; i++) {
      const rlPoint = rlData?.trajectory[i];
      const esPoint = esData?.trajectory[i];

      if (rlPoint) rlCum += rlPoint.reward;
      if (esPoint) esCum += esPoint.reward;

      data.push({
        t: (rlPoint?.t ?? esPoint?.t ?? i * 0.1).toFixed(1),
        rlInstant: rlPoint?.reward ?? null,
        esInstant: esPoint?.reward ?? null,
        rlCumulative: rlPoint ? rlCum : null,
        esCumulative: esPoint ? esCum : null,
      });
    }

    return data;
  }, [rlData, esData]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card/90 backdrop-blur-sm border border-muted rounded-lg p-3 shadow-xl">
          <p className="text-xs text-muted-foreground mb-2">Time: {label}s</p>
          {payload.map((entry: any, index: number) => (
            <p
              key={index}
              className="text-sm font-mono"
              style={{ color: entry.color }}
            >
              {entry.name}: {entry.value?.toFixed(2) ?? 'N/A'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Find current data index for reference line
  const currentIndex = chartData.findIndex(
    (d) => parseFloat(d.t) >= currentTime
  );

  return (
    <div className="bg-card/30 backdrop-blur-sm rounded-xl border border-muted p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
          Reward Comparison
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Instant</span>
          <Switch checked={showCumulative} onCheckedChange={setShowCumulative} />
          <span className="text-xs text-muted-foreground">Cumulative</span>
        </div>
      </div>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis
              dataKey="t"
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              tickFormatter={(v) => `${v}s`}
            />
            <YAxis
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              width={40}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '10px' }}
              formatter={(value) => (
                <span className="text-xs font-mono">{value}</span>
              )}
            />
            {showCumulative ? (
              <>
                <Line
                  type="monotone"
                  dataKey="rlCumulative"
                  name="RL Cumulative"
                  stroke="#00d4ff"
                  strokeWidth={2}
                  dot={false}
                  filter="drop-shadow(0 0 4px #00d4ff)"
                />
                <Line
                  type="monotone"
                  dataKey="esCumulative"
                  name="ES Cumulative"
                  stroke="#ff3366"
                  strokeWidth={2}
                  dot={false}
                  filter="drop-shadow(0 0 4px #ff3366)"
                />
              </>
            ) : (
              <>
                <Line
                  type="monotone"
                  dataKey="rlInstant"
                  name="RL Instant"
                  stroke="#00d4ff"
                  strokeWidth={2}
                  dot={false}
                  filter="drop-shadow(0 0 4px #00d4ff)"
                />
                <Line
                  type="monotone"
                  dataKey="esInstant"
                  name="ES Instant"
                  stroke="#ff3366"
                  strokeWidth={2}
                  dot={false}
                  filter="drop-shadow(0 0 4px #ff3366)"
                />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
