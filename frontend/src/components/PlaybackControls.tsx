import { Play, Pause, RotateCcw, Repeat } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { PlaybackSpeed } from '@/types/race';

interface PlaybackControlsProps {
  isPlaying: boolean;
  currentTime: number;
  maxTime: number;
  playbackSpeed: PlaybackSpeed;
  loop: boolean;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onSeek: (time: number) => void;
  onSpeedChange: (speed: PlaybackSpeed) => void;
  onLoopChange: (loop: boolean) => void;
}

const speeds: PlaybackSpeed[] = [1, 2, 5, 10];

export const PlaybackControls = ({
  isPlaying,
  currentTime,
  maxTime,
  playbackSpeed,
  loop,
  onPlay,
  onPause,
  onReset,
  onSeek,
  onSpeedChange,
  onLoopChange,
}: PlaybackControlsProps) => {
  const formatTime = (t: number) => {
    const mins = Math.floor(t / 60);
    const secs = (t % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
  };

  return (
    <div className="bg-card/50 backdrop-blur-md rounded-xl border border-muted p-4 space-y-4">
      <div className="flex items-center gap-4">
        {/* Play/Pause */}
        <Button
          variant="ghost"
          size="icon"
          onClick={isPlaying ? onPause : onPlay}
          className="h-12 w-12 rounded-full bg-primary/20 hover:bg-primary/30 text-primary border border-primary/30"
        >
          {isPlaying ? (
            <Pause className="h-6 w-6" />
          ) : (
            <Play className="h-6 w-6 ml-0.5" />
          )}
        </Button>

        {/* Reset */}
        <Button
          variant="ghost"
          size="icon"
          onClick={onReset}
          className="h-10 w-10 rounded-full hover:bg-muted text-muted-foreground"
        >
          <RotateCcw className="h-5 w-5" />
        </Button>

        {/* Time display */}
        <div className="flex-1 flex items-center gap-3">
          <span className="text-sm font-mono text-primary min-w-[60px]">
            {formatTime(currentTime)}
          </span>
          <Slider
            value={[currentTime]}
            max={maxTime}
            step={0.1}
            onValueChange={([value]) => onSeek(value)}
            className="flex-1"
          />
          <span className="text-sm font-mono text-muted-foreground min-w-[60px]">
            {formatTime(maxTime)}
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between">
        {/* Speed controls */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground mr-2">SPEED</span>
          {speeds.map((speed) => (
            <Button
              key={speed}
              variant="ghost"
              size="sm"
              onClick={() => onSpeedChange(speed)}
              className={`h-8 px-3 font-mono text-xs ${
                playbackSpeed === speed
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {speed}x
            </Button>
          ))}
        </div>

        {/* Loop toggle */}
        <div className="flex items-center gap-2">
          <Repeat
            className={`h-4 w-4 ${loop ? 'text-primary' : 'text-muted-foreground'}`}
          />
          <span className="text-xs text-muted-foreground">LOOP</span>
          <Switch checked={loop} onCheckedChange={onLoopChange} />
        </div>
      </div>
    </div>
  );
};
