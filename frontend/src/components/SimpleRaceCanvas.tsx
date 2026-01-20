
import { useRef, useEffect, useMemo } from 'react';
import { AgentState } from '@/types/race';

interface SimpleRaceCanvasProps {
    trackPoints: [number, number][]; // specific boundary box
    rlState: AgentState | null;
    esState: AgentState | null;
    rlTrail: { x: number; y: number }[];
    esTrail: { x: number; y: number }[];
    width?: number;
    height?: number;
}

export const SimpleRaceCanvas = ({
    trackPoints,
    rlState,
    esState,
    rlTrail,
    esTrail,
}: SimpleRaceCanvasProps) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Quick-calculate transform matrix (memoized)
    const transform = useMemo(() => {
        // If no points, standard 100x100 box
        let minX = 0, maxX = 100, minY = 0, maxY = 50;

        if (trackPoints && trackPoints.length > 0) {
            const xs = trackPoints.map(p => p[0]);
            const ys = trackPoints.map(p => p[1]);
            minX = Math.min(...xs);
            maxX = Math.max(...xs);
            minY = Math.min(...ys);
            maxY = Math.max(...ys);
        }

        const padding = 20; // Less padding for "zoom" feel
        const width = 800;  // Internal resolution
        const height = 400; // Internal resolution

        const dataW = maxX - minX || 1;
        const dataH = maxY - minY || 1;

        const scaleX = (width - padding * 2) / dataW;
        const scaleY = (height - padding * 2) / dataH;
        const scale = Math.min(scaleX, scaleY);

        const offsetX = (width - dataW * scale) / 2 - minX * scale;
        const offsetY = (height - dataH * scale) / 2 - minY * scale;

        return { scale, offsetX, offsetY };
    }, [trackPoints]);

    const toCanvas = (x: number, y: number) => ({
        x: x * transform.scale + transform.offsetX,
        y: y * transform.scale + transform.offsetY,
    });

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d', { alpha: false }); // Optimize for no transparency if possible
        if (!ctx) return;

        // Set high internal resolution
        canvas.width = 800;
        canvas.height = 400;

        // 1. Clear (Solid Background for speed)
        ctx.fillStyle = '#0f172a'; // dark-slate-900
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 2. Draw Grid (Simple faint lines)
        ctx.strokeStyle = '#1e293b'; // slate-800
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x <= canvas.width; x += 50) { ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); }
        for (let y = 0; y <= canvas.height; y += 50) { ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); }
        ctx.stroke();

        // 3. Draw Trails (Single Path = Fast)
        const drawSimpleTrail = (trail: { x: number, y: number }[], color: string) => {
            if (trail.length < 2) return;
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.beginPath();
            const start = toCanvas(trail[0].x, trail[0].y);
            ctx.moveTo(start.x, start.y);
            for (let i = 1; i < trail.length; i++) {
                const p = toCanvas(trail[i].x, trail[i].y);
                ctx.lineTo(p.x, p.y);
            }
            ctx.stroke();
        };

        drawSimpleTrail(rlTrail, '#3b82f6'); // Blue-500
        drawSimpleTrail(esTrail, '#ef4444'); // Red-500

        // 4. Draw Agents (Simple Triangle)
        const drawAgent = (state: AgentState | null, color: string) => {
            if (!state) return;
            const p = toCanvas(state.x, state.y);
            const size = 12;

            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(state.heading);

            ctx.fillStyle = color;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;

            ctx.beginPath();
            ctx.moveTo(size, 0); // Front
            ctx.lineTo(-size / 2, -size / 2); // Back Left
            ctx.lineTo(-size / 2, size / 2); // Back Right
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            ctx.restore();
        };

        drawAgent(rlState, '#3b82f6');
        drawAgent(esState, '#ef4444');

    }, [rlState, esState, rlTrail, esTrail, transform]);

    return (
        <div className="w-full h-full flex items-center justify-center bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
            <canvas
                ref={canvasRef}
                className="w-full h-auto object-contain"
            />
        </div>
    );
};
