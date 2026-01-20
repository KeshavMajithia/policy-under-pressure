import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Shield, Zap, Target, Trophy } from 'lucide-react';
import { RaceData } from '@/types/race';

interface AgentBadgesProps {
  rlData: RaceData | null;
  esData: RaceData | null;
}

interface Badge {
  icon: React.ReactNode;
  name: string;
  description: string;
  winner: 'RL' | 'ES' | 'BOTH' | null;
}

export const AgentBadges = ({ rlData, esData }: AgentBadgesProps) => {
  const badges = useMemo<Badge[]>(() => {
    if (!rlData || !esData) return [];

    const rlTraj = rlData.trajectory;
    const esTraj = esData.trajectory;

    // Survivor: Award if trajectory length equals maximum expected
    const rlSurvivor = rlTraj.length >= 50;
    const esSurvivor = esTraj.length >= 50;

    // Speed Demon: Highest max speed
    const rlMaxSpeed = Math.max(...rlTraj.map((p) => p.speed));
    const esMaxSpeed = Math.max(...esTraj.map((p) => p.speed));
    const speedWinner = rlMaxSpeed > esMaxSpeed ? 'RL' : esMaxSpeed > rlMaxSpeed ? 'ES' : 'BOTH';

    // Precision: Lowest steering std deviation
    const calcStdDev = (values: number[]) => {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
      return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / values.length);
    };

    const rlSteeringStd = calcStdDev(rlTraj.map((p) => p.steering));
    const esSteeringStd = calcStdDev(esTraj.map((p) => p.steering));
    const precisionWinner =
      rlSteeringStd < esSteeringStd ? 'RL' : esSteeringStd < rlSteeringStd ? 'ES' : 'BOTH';

    // Total Reward Winner
    const rewardWinner =
      rlData.metadata.total_reward > esData.metadata.total_reward
        ? 'RL'
        : esData.metadata.total_reward > rlData.metadata.total_reward
        ? 'ES'
        : 'BOTH';

    return [
      {
        icon: <Shield className="w-5 h-5" />,
        name: 'Survivor',
        description: 'Completed full run without crash',
        winner: rlSurvivor && esSurvivor ? 'BOTH' : rlSurvivor ? 'RL' : esSurvivor ? 'ES' : null,
      },
      {
        icon: <Zap className="w-5 h-5" />,
        name: 'Speed Demon',
        description: `Max: RL ${rlMaxSpeed.toFixed(1)} / ES ${esMaxSpeed.toFixed(1)} m/s`,
        winner: speedWinner,
      },
      {
        icon: <Target className="w-5 h-5" />,
        name: 'Precision',
        description: 'Lowest steering variance',
        winner: precisionWinner,
      },
      {
        icon: <Trophy className="w-5 h-5" />,
        name: 'Champion',
        description: `Total: RL ${rlData.metadata.total_reward.toFixed(1)} / ES ${esData.metadata.total_reward.toFixed(1)}`,
        winner: rewardWinner,
      },
    ];
  }, [rlData, esData]);

  if (!rlData || !esData) return null;

  const getWinnerColor = (winner: 'RL' | 'ES' | 'BOTH' | null) => {
    switch (winner) {
      case 'RL':
        return '#00d4ff';
      case 'ES':
        return '#ff3366';
      case 'BOTH':
        return '#a855f7';
      default:
        return '#64748b';
    }
  };

  const getWinnerLabel = (winner: 'RL' | 'ES' | 'BOTH' | null) => {
    switch (winner) {
      case 'RL':
        return 'RL Wins';
      case 'ES':
        return 'ES Wins';
      case 'BOTH':
        return 'Tie';
      default:
        return 'N/A';
    }
  };

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
        Achievements
      </h3>
      <div className="grid grid-cols-2 gap-2">
        {badges.map((badge, index) => (
          <motion.div
            key={badge.name}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-card/30 backdrop-blur-sm rounded-lg border border-muted p-3 relative overflow-hidden"
          >
            {/* Glow effect */}
            <div
              className="absolute inset-0 opacity-10"
              style={{
                background: `radial-gradient(circle at 30% 30%, ${getWinnerColor(
                  badge.winner
                )}, transparent 70%)`,
              }}
            />

            <div className="relative flex items-start gap-3">
              <div
                className="p-2 rounded-lg"
                style={{
                  backgroundColor: `${getWinnerColor(badge.winner)}20`,
                  color: getWinnerColor(badge.winner),
                  boxShadow: `0 0 15px ${getWinnerColor(badge.winner)}40`,
                }}
              >
                {badge.icon}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold text-foreground">
                    {badge.name}
                  </span>
                  <span
                    className="text-[10px] font-mono px-2 py-0.5 rounded-full"
                    style={{
                      backgroundColor: `${getWinnerColor(badge.winner)}20`,
                      color: getWinnerColor(badge.winner),
                    }}
                  >
                    {getWinnerLabel(badge.winner)}
                  </span>
                </div>
                <p className="text-[10px] text-muted-foreground mt-1 truncate">
                  {badge.description}
                </p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};
