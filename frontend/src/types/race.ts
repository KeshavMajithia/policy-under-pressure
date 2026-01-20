export interface TrajectoryPoint {
  t: number;
  x: number;
  y: number;
  heading: number;
  speed: number;
  steering: number;
  throttle: number;
  reward: number;
}

export interface RaceMetadata {
  agent_type: "RL" | "ES";
  algorithm: string;
  track_points: [number, number][];
  total_time: number;
  total_reward: number;
}

export interface RaceData {
  metadata: RaceMetadata;
  trajectory: TrajectoryPoint[];
}

export interface AgentState {
  x: number;
  y: number;
  heading: number;
  speed: number;
  steering: number;
  throttle: number;
  reward: number;
  cumulativeReward: number;
}

export type PlaybackSpeed = 1 | 2 | 5 | 10;
