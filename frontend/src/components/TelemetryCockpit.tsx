import { AgentState } from '@/types/race';
import { motion } from 'framer-motion';

interface TelemetryCockpitProps {
  rlState: AgentState | null;
  esState: AgentState | null;
}

const SteeringWheel = ({ steering, color }: { steering: number; color: string }) => {
  const rotation = steering * 90; // -90 to 90 degrees

  return (
    <div className="relative w-20 h-20">
      <svg viewBox="0 0 100 100" className="w-full h-full">
        {/* Outer ring */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="currentColor"
          strokeWidth="4"
          className="text-muted"
        />
        {/* Wheel spokes */}
        <motion.g
          animate={{ rotate: rotation }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          style={{ transformOrigin: '50px 50px' }}
        >
          <circle
            cx="50"
            cy="50"
            r="40"
            fill="none"
            stroke={color}
            strokeWidth="6"
            className="drop-shadow-lg"
            style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          />
          <line
            x1="50"
            y1="15"
            x2="50"
            y2="50"
            stroke={color}
            strokeWidth="4"
            strokeLinecap="round"
          />
          <line
            x1="15"
            y1="60"
            x2="50"
            y2="50"
            stroke={color}
            strokeWidth="4"
            strokeLinecap="round"
          />
          <line
            x1="85"
            y1="60"
            x2="50"
            y2="50"
            stroke={color}
            strokeWidth="4"
            strokeLinecap="round"
          />
          <circle cx="50" cy="50" r="8" fill={color} />
        </motion.g>
      </svg>
    </div>
  );
};

const VerticalGauge = ({
  value,
  label,
  color,
  maxValue = 1,
}: {
  value: number;
  label: string;
  color: string;
  maxValue?: number;
}) => {
  const percentage = Math.min((value / maxValue) * 100, 100);

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative w-8 h-20 bg-muted rounded-lg overflow-hidden border border-muted-foreground/20">
        <motion.div
          className="absolute bottom-0 left-0 right-0 rounded-b-lg"
          style={{ backgroundColor: color, boxShadow: `0 0 10px ${color}` }}
          initial={{ height: 0 }}
          animate={{ height: `${percentage}%` }}
          transition={{ type: 'spring', stiffness: 200, damping: 20 }}
        />
        {/* Graduation marks */}
        {[25, 50, 75].map((mark) => (
          <div
            key={mark}
            className="absolute left-0 right-0 h-px bg-muted-foreground/30"
            style={{ bottom: `${mark}%` }}
          />
        ))}
      </div>
      <span className="text-[10px] font-mono text-muted-foreground uppercase">
        {label}
      </span>
    </div>
  );
};

const SpeedDisplay = ({ speed, color }: { speed: number; color: string }) => (
  <div className="text-center">
    <motion.div
      className="text-3xl font-bold font-mono"
      style={{ color, textShadow: `0 0 20px ${color}` }}
      key={Math.round(speed)}
      initial={{ scale: 1.1 }}
      animate={{ scale: 1 }}
      transition={{ duration: 0.1 }}
    >
      {Math.round(speed)}
    </motion.div>
    <div className="text-[10px] text-muted-foreground font-mono">M/S</div>
  </div>
);

const AgentTelemetry = ({
  state,
  label,
  color,
}: {
  state: AgentState | null;
  label: string;
  color: string;
}) => {
  if (!state) {
    return (
      <div className="flex-1 bg-card/30 rounded-xl border border-muted p-4 opacity-50">
        <div className="text-center text-muted-foreground">No Data</div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-card/30 backdrop-blur-sm rounded-xl border border-muted p-4">
      <div
        className="text-xs font-mono font-bold mb-3 text-center"
        style={{ color }}
      >
        {label}
      </div>

      <div className="flex items-center justify-around">
        <SteeringWheel steering={state.steering} color={color} />
        <SpeedDisplay speed={state.speed} color={color} />
        <div className="flex gap-3">
          <VerticalGauge
            value={state.throttle}
            label="THR"
            color="#22c55e"
          />
          <VerticalGauge
            value={Math.abs(state.steering)}
            label="STR"
            color={color}
          />
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-muted grid grid-cols-2 gap-2 text-xs font-mono">
        <div className="text-muted-foreground">
          POS: <span className="text-foreground">{state.x.toFixed(1)}, {state.y.toFixed(1)}</span>
        </div>
        <div className="text-muted-foreground">
          HDG: <span className="text-foreground">{((state.heading * 180) / Math.PI).toFixed(1)}°</span>
        </div>
        <div className="text-muted-foreground">
          REW: <span style={{ color }}>{state.reward.toFixed(2)}</span>
        </div>
        <div className="text-muted-foreground">
          CUM: <span style={{ color }}>{state.cumulativeReward.toFixed(1)}</span>
        </div>
      </div>
    </div>
  );
};

export const TelemetryCockpit = ({ rlState, esState }: TelemetryCockpitProps) => {
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
        Live Telemetry
      </h3>
      <div className="flex gap-3">
        <AgentTelemetry state={rlState} label="RL AGENT (PPO)" color="#00d4ff" />
        <AgentTelemetry state={esState} label="ES AGENT" color="#ff3366" />
      </div>
    </div>
  );
};
