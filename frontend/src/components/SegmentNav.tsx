import { Link, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';

export const SegmentNav = () => {
    const location = useLocation();
    const currentPath = location.pathname;

    const segments = [
        { path: '/segment1', label: 'SEG 1', fullLabel: 'SEGMENT 1', color: 'primary' },
        { path: '/segment2', label: 'SEG 2', fullLabel: 'SEGMENT 2', color: 'secondary' },
        { path: '/segment3', label: 'SEG 3', fullLabel: 'SEGMENT 3', color: 'white' },
        { path: '/segment4', label: 'SEG 4', fullLabel: 'SEGMENT 4', color: 'amber-500' },
    ];

    return (
        <div className="flex items-center gap-1 sm:gap-2 bg-background/50 p-1 rounded-lg border border-muted overflow-x-auto">
            {segments.map((segment) => {
                const isActive = currentPath === segment.path;
                const colorClasses = {
                    'primary': isActive
                        ? 'bg-primary/20 text-primary border-primary/40'
                        : 'bg-primary/5 hover:bg-primary/10 text-primary/60 border-primary/10',
                    'secondary': isActive
                        ? 'bg-secondary/20 text-secondary border-secondary/40'
                        : 'bg-secondary/5 hover:bg-secondary/10 text-secondary/60 border-secondary/10',
                    'white': isActive
                        ? 'bg-white/20 text-white border-white/40'
                        : 'bg-white/5 hover:bg-white/10 text-white/60 border-white/10',
                    'amber-500': isActive
                        ? 'bg-amber-500/20 text-amber-500 border-amber-500/40'
                        : 'bg-amber-500/5 hover:bg-amber-500/10 text-amber-500/60 border-amber-500/10',
                };

                return (
                    <Link
                        key={segment.path}
                        to={segment.path}
                        className={`px-2 sm:px-3 py-1.5 text-xs font-mono font-bold rounded-md transition-all flex items-center gap-1 sm:gap-2 border whitespace-nowrap ${colorClasses[segment.color]}`}
                    >
                        <Activity className="w-3 h-3 flex-shrink-0" />
                        <span className="hidden sm:inline">{segment.fullLabel}</span>
                        <span className="sm:hidden">{segment.label}</span>
                    </Link>
                );
            })}
        </div>
    );
};
