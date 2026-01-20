import { Link, useLocation } from 'react-router-dom';
import { Activity, ArrowLeft, Cpu } from 'lucide-react';

interface SegmentHeaderProps {
    title: string;
    subtitle?: string;
    neonText: string;
    color: 'primary' | 'secondary' | 'white' | 'amber';
}

export const SegmentHeader = ({ title, subtitle, neonText, color }: SegmentHeaderProps) => {
    const location = useLocation();
    const currentPath = location.pathname;

    const segments = [
        { path: '/segment1', label: 'S1', color: 'primary', borderColor: 'border-primary/20', bgColor: 'bg-primary/10', hoverBg: 'hover:bg-primary/20', textColor: 'text-primary' },
        { path: '/segment2', label: 'S2', color: 'secondary', borderColor: 'border-secondary/20', bgColor: 'bg-secondary/10', hoverBg: 'hover:bg-secondary/20', textColor: 'text-secondary' },
        { path: '/segment3', label: 'S3', color: 'white', borderColor: 'border-white/20', bgColor: 'bg-white/10', hoverBg: 'hover:bg-white/20', textColor: 'text-white' },
        { path: '/segment4', label: 'S4', color: 'amber', borderColor: 'border-amber-500/20', bgColor: 'bg-amber-500/10', hoverBg: 'hover:bg-amber-500/20', textColor: 'text-amber-500' },
    ];

    const neonColors = {
        primary: 'neon-text-primary',
        secondary: 'neon-text-secondary',
        white: 'neon-text-white',
        amber: 'neon-text-amber'
    };

    const iconColors = {
        primary: 'text-primary bg-primary/20',
        secondary: 'text-secondary bg-secondary/20',
        white: 'text-white bg-white/20',
        amber: 'text-amber-500 bg-amber-500/20'
    };

    return (
        <header className="sticky top-0 z-50 border-b border-muted bg-card/80 backdrop-blur-md">
            <div className="container mx-auto px-4 py-3 sm:py-4">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
                    {/* Title Section with Back Button */}
                    <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
                        <Link to="/" className="p-1.5 sm:p-2 hover:bg-muted rounded-full transition-colors flex-shrink-0">
                            <ArrowLeft className="w-4 h-4 sm:w-5 sm:h-5 text-muted-foreground" />
                        </Link>
                        <div className={`p-1.5 sm:p-2 rounded-lg ${iconColors[color]} cyber-glow flex-shrink-0`}>
                            <Cpu className="w-4 h-4 sm:w-5 sm:h-5" />
                        </div>
                        <div className="min-w-0 flex-1">
                            <h1 className="text-sm sm:text-lg font-display font-bold text-foreground tracking-wider truncate">
                                {title} <span className={`${neonColors[color]} neon-text`}>{neonText}</span>
                            </h1>
                            {subtitle && (
                                <p className="text-xs text-muted-foreground font-mono hidden sm:block truncate">
                                    {subtitle}
                                </p>
                            )}
                        </div>
                    </div>

                    {/* Navigation Buttons */}
                    <div className="flex items-center gap-1 sm:gap-2 bg-background/50 p-1 rounded-lg border border-muted w-full sm:w-auto">
                        {segments.map((segment) => {
                            const isActive = currentPath === segment.path;
                            return (
                                <Link
                                    key={segment.path}
                                    to={segment.path}
                                    className={`
                    px-3 sm:px-4 py-2 text-xs font-mono font-bold rounded-md transition-all 
                    flex items-center gap-1.5 border flex-1 sm:flex-none justify-center
                    ${isActive
                                            ? `${segment.bgColor} ${segment.textColor} ${segment.borderColor}`
                                            : `${segment.bgColor} ${segment.textColor} ${segment.borderColor} ${segment.hoverBg} opacity-60 hover:opacity-100`
                                        }
                  `}
                                >
                                    <Activity className="w-3 h-3" />
                                    {segment.label}
                                </Link>
                            );
                        })}
                    </div>
                </div>

                {/* Status Indicators - Only on Desktop */}
                <div className="hidden sm:flex items-center gap-4 mt-2 ml-auto w-fit">
                    <div className="flex items-center gap-2">
                        <Activity className="w-3 h-3 text-primary animate-pulse" />
                        <span className="text-xs font-mono text-muted-foreground">LIVE</span>
                    </div>
                    <div className="flex items-center gap-3 text-xs font-mono">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-rl shadow-lg shadow-rl/50 animate-pulse" />
                            <span className="text-rl">RL</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-es shadow-lg shadow-es/50 animate-pulse" />
                            <span className="text-es">ES</span>
                        </div>
                    </div>
                </div>
            </div>
        </header>
    );
};
