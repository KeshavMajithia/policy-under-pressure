import { useRef, useEffect, useMemo } from 'react';
import { RaceData, AgentState } from '@/types/race';

interface RaceCanvasProps {
  trackPoints: [number, number][];
  rlState: AgentState | null;
  esState: AgentState | null;
  rlTrail: { x: number; y: number }[];
  esTrail: { x: number; y: number }[];
}

export const RaceCanvas = ({
  trackPoints,
  rlState,
  esState,
  rlTrail,
  esTrail,
}: RaceCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate bounds and transform
  const transform = useMemo(() => {
    if (trackPoints.length === 0) return { scale: 1, offsetX: 0, offsetY: 0 };

    const xs = trackPoints.map((p) => p[0]);
    const ys = trackPoints.map((p) => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const padding = 60;
    const width = 800;
    const height = 500;

    const scaleX = (width - padding * 2) / (maxX - minX || 1);
    const scaleY = (height - padding * 2) / (maxY - minY || 1);
    const scale = Math.min(scaleX, scaleY);

    const offsetX = (width - (maxX - minX) * scale) / 2 - minX * scale;
    const offsetY = (height - (maxY - minY) * scale) / 2 - minY * scale;

    return { scale, offsetX, offsetY };
  }, [trackPoints]);

  const toCanvas = (x: number, y: number) => ({
    x: x * transform.scale + transform.offsetX,
    y: y * transform.scale + transform.offsetY,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = 800;
    canvas.height = 500;

    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw track with glow
    if (trackPoints.length > 1) {
      // Outer glow
      ctx.shadowColor = '#a855f7';
      ctx.shadowBlur = 20;
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.3)';
      ctx.lineWidth = 30;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      const first = toCanvas(trackPoints[0][0], trackPoints[0][1]);
      ctx.moveTo(first.x, first.y);
      trackPoints.slice(1).forEach((point) => {
        const p = toCanvas(point[0], point[1]);
        ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();

      // Main track
      ctx.shadowBlur = 15;
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.6)';
      ctx.lineWidth = 16;
      ctx.beginPath();
      ctx.moveTo(first.x, first.y);
      trackPoints.slice(1).forEach((point) => {
        const p = toCanvas(point[0], point[1]);
        ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();

      // Inner track
      ctx.shadowBlur = 8;
      ctx.strokeStyle = '#c084fc';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(first.x, first.y);
      trackPoints.slice(1).forEach((point) => {
        const p = toCanvas(point[0], point[1]);
        ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Draw trails with gradient fade
    const drawTrail = (
      trail: { x: number; y: number }[],
      color: string,
      glowColor: string
    ) => {
      if (trail.length < 2) return;

      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (let i = 1; i < trail.length; i++) {
        const alpha = i / trail.length;
        ctx.strokeStyle = `${color}${Math.floor(alpha * 255).toString(16).padStart(2, '0')}`;
        ctx.lineWidth = 3 + alpha * 3;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 10 * alpha;

        const from = toCanvas(trail[i - 1].x, trail[i - 1].y);
        const to = toCanvas(trail[i].x, trail[i].y);

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
    };

    drawTrail(rlTrail, '#00d4ff', '#00d4ff');
    drawTrail(esTrail, '#ff3366', '#ff3366');

    // Draw car function
    const drawCar = (
      state: AgentState,
      primaryColor: string,
      glowColor: string,
      label: string
    ) => {
      const pos = toCanvas(state.x, state.y);
      const size = 18;

      ctx.save();
      ctx.translate(pos.x, pos.y);
      ctx.rotate(state.heading);

      // Glow effect
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 20;

      // Car body
      ctx.fillStyle = primaryColor;
      ctx.beginPath();
      ctx.moveTo(size, 0);
      ctx.lineTo(-size * 0.6, -size * 0.5);
      ctx.lineTo(-size * 0.4, 0);
      ctx.lineTo(-size * 0.6, size * 0.5);
      ctx.closePath();
      ctx.fill();

      // Cockpit
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.beginPath();
      ctx.ellipse(size * 0.2, 0, size * 0.3, size * 0.2, 0, 0, Math.PI * 2);
      ctx.fill();

      // Headlight
      ctx.fillStyle = '#ffffff';
      ctx.shadowColor = '#ffffff';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(size * 0.8, 0, 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();

      // Label
      ctx.shadowBlur = 0;
      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = primaryColor;
      ctx.textAlign = 'center';
      ctx.fillText(label, pos.x, pos.y - 25);
    };

    // Draw cars
    if (rlState) {
      drawCar(rlState, '#00d4ff', '#00d4ff', 'RL (PPO)');
    }

    if (esState) {
      drawCar(esState, '#ff3366', '#ff3366', 'ES');
    }

    // Draw start/finish marker
    const start = toCanvas(trackPoints[0][0], trackPoints[0][1]);
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 4;
    ctx.shadowColor = '#22c55e';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.moveTo(start.x - 15, start.y - 20);
    ctx.lineTo(start.x - 15, start.y + 20);
    ctx.stroke();

    // Checkered pattern
    ctx.shadowBlur = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 2; j++) {
        ctx.fillStyle = (i + j) % 2 === 0 ? '#ffffff' : '#000000';
        ctx.fillRect(start.x - 15 + j * 8, start.y - 20 + i * 10, 8, 10);
      }
    }
  }, [trackPoints, rlState, esState, rlTrail, esTrail, transform]);

  return (
    <div ref={containerRef} className="relative w-full h-full flex items-center justify-center">
      <canvas
        ref={canvasRef}
        className="rounded-xl border-2 border-track/30 shadow-2xl shadow-track/20"
        style={{ maxWidth: '100%', height: 'auto' }}
      />
      
      {/* Legend */}
      <div className="absolute top-4 right-4 bg-background/80 backdrop-blur-sm rounded-lg p-3 border border-muted">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-3 h-3 rounded-full bg-rl shadow-lg shadow-rl/50" />
          <span className="text-xs text-rl font-mono">RL Agent (PPO)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-es shadow-lg shadow-es/50" />
          <span className="text-xs text-es font-mono">ES Agent</span>
        </div>
      </div>
    </div>
  );
};
