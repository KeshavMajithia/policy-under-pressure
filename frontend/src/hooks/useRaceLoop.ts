import { useState, useRef, useCallback, useEffect } from 'react';
import { RaceData, AgentState, PlaybackSpeed, TrajectoryPoint } from '@/types/race';

// Linear interpolation
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

// Angle interpolation (handles wraparound)
const lerpAngle = (a: number, b: number, t: number) => {
  let diff = b - a;
  while (diff > Math.PI) diff -= 2 * Math.PI;
  while (diff < -Math.PI) diff += 2 * Math.PI;
  return a + diff * t;
};

const interpolateState = (
  prev: TrajectoryPoint,
  next: TrajectoryPoint,
  alpha: number,
  cumulativeReward: number
): AgentState => ({
  x: lerp(prev.x, next.x, alpha),
  y: lerp(prev.y, next.y, alpha),
  heading: lerpAngle(prev.heading, next.heading, alpha),
  speed: lerp(prev.speed, next.speed, alpha),
  steering: lerp(prev.steering, next.steering, alpha),
  throttle: lerp(prev.throttle, next.throttle, alpha),
  reward: lerp(prev.reward, next.reward, alpha),
  cumulativeReward,
});

export const useRaceLoop = (rlData: RaceData | null, esData: RaceData | null) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<PlaybackSpeed>(1);
  const [loop, setLoop] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [rlState, setRlState] = useState<AgentState | null>(null);
  const [esState, setEsState] = useState<AgentState | null>(null);
  const [rlTrail, setRlTrail] = useState<{ x: number; y: number }[]>([]);
  const [esTrail, setEsTrail] = useState<{ x: number; y: number }[]>([]);

  const lastTimeRef = useRef<number>(0);
  const animationRef = useRef<number>(0);

  const maxTime = Math.max(
    rlData?.metadata.total_time ?? 0,
    esData?.metadata.total_time ?? 0
  );

  const getStateAtTime = useCallback(
    (data: RaceData, time: number): { state: AgentState; cumulative: number } | null => {
      const trajectory = data.trajectory;
      if (trajectory.length === 0) return null;

      // Clamp time to trajectory bounds
      const clampedTime = Math.min(time, trajectory[trajectory.length - 1].t);

      // Find surrounding keyframes
      let prevIdx = 0;
      for (let i = 0; i < trajectory.length - 1; i++) {
        if (trajectory[i + 1].t > clampedTime) {
          prevIdx = i;
          break;
        }
        prevIdx = i;
      }

      const prev = trajectory[prevIdx];
      const next = trajectory[Math.min(prevIdx + 1, trajectory.length - 1)];

      // Calculate alpha for interpolation
      const dt = next.t - prev.t;
      const alpha = dt > 0 ? (clampedTime - prev.t) / dt : 0;

      // Calculate cumulative reward up to this point
      let cumulative = 0;
      for (let i = 0; i <= prevIdx; i++) {
        cumulative += trajectory[i].reward;
      }
      cumulative += prev.reward * alpha;

      return {
        state: interpolateState(prev, next, alpha, cumulative),
        cumulative,
      };
    },
    []
  );

  const updateStates = useCallback(
    (time: number) => {
      if (rlData) {
        const result = getStateAtTime(rlData, time);
        if (result) {
          setRlState(result.state);
          setRlTrail((prev) => {
            const newTrail = [...prev, { x: result.state.x, y: result.state.y }];
            return newTrail.slice(-80); // Keep last 80 points
          });
        }
      }

      if (esData) {
        const result = getStateAtTime(esData, time);
        if (result) {
          setEsState(result.state);
          setEsTrail((prev) => {
            const newTrail = [...prev, { x: result.state.x, y: result.state.y }];
            return newTrail.slice(-80);
          });
        }
      }
    },
    [rlData, esData, getStateAtTime]
  );

  const animate = useCallback(
    (timestamp: number) => {
      if (!lastTimeRef.current) {
        lastTimeRef.current = timestamp;
      }

      const deltaMs = timestamp - lastTimeRef.current;
      lastTimeRef.current = timestamp;

      const deltaSimTime = (deltaMs / 1000) * playbackSpeed;

      setCurrentTime((prev) => {
        let newTime = prev + deltaSimTime;

        if (newTime >= maxTime) {
          if (loop) {
            setRlTrail([]);
            setEsTrail([]);
            return 0;
          } else {
            setIsPlaying(false);
            return maxTime;
          }
        }

        return newTime;
      });

      animationRef.current = requestAnimationFrame(animate);
    },
    [playbackSpeed, maxTime, loop]
  );

  useEffect(() => {
    updateStates(currentTime);
  }, [currentTime, updateStates]);

  // Reset when data changes (e.g. scenario switch)
  useEffect(() => {
    setCurrentTime(0);
    setRlTrail([]);
    setEsTrail([]);
    setRlState(null);
    setEsState(null);
  }, [rlData, esData]);

  useEffect(() => {
    if (isPlaying) {
      lastTimeRef.current = 0;
      animationRef.current = requestAnimationFrame(animate);
    } else {
      cancelAnimationFrame(animationRef.current);
    }

    return () => cancelAnimationFrame(animationRef.current);
  }, [isPlaying, animate]);

  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);
  const reset = useCallback(() => {
    setCurrentTime(0);
    setRlTrail([]);
    setEsTrail([]);
  }, []);

  const seek = useCallback((time: number) => {
    setCurrentTime(Math.max(0, Math.min(time, maxTime)));
    setRlTrail([]);
    setEsTrail([]);
  }, [maxTime]);

  return {
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
  };
};
