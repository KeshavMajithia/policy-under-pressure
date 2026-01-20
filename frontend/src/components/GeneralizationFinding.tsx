import React from 'react';
import { Network, GitGraph } from 'lucide-react';

export const GeneralizationFinding = () => {
    return (
        <div className="bg-card/40 backdrop-blur-md rounded-xl border border-muted p-6 relative overflow-hidden group hover:border-primary/50 transition-colors">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <Network className="w-24 h-24 text-primary" />
            </div>

            <div className="flex items-start gap-4 relative z-10">
                <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                    <GitGraph className="w-6 h-6 text-purple-400" />
                </div>

                <div className="space-y-4 flex-1">
                    <div>
                        <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2">
                            FINDING #3: GENERALIZATION <span className="text-xs font-mono text-purple-400 px-2 py-0.5 rounded bg-purple-500/10">DIMENSION 4</span>
                        </h3>
                        <p className="text-sm text-muted-foreground mt-1 font-mono">
                            Tested on: Figure-8 Track (Complex Topology)
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <h4 className="text-xs font-bold text-foreground uppercase tracking-wider border-l-2 border-rl pl-2">
                                Reinforcement Learning (PPO)
                            </h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                With <span className="text-rl">10x training compute</span> (300k steps), the RL agent successfully mastered the complex Figure-8 topology. It learned to anticipate the crossover and alternating curves without any driver assists, demonstrating true <strong>zero-shot generalization</strong> to unseen geometries when given sufficient training depth.
                            </p>
                        </div>

                        <div className="space-y-2">
                            <h4 className="text-xs font-bold text-foreground uppercase tracking-wider border-l-2 border-es pl-2">
                                Evolution Strategies (ES)
                            </h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                ES struggled to converge even with increased population budget (100 generations). The lack of gradient information makes traversing the specific "crossover" point difficult, as the reward landscape is sparse. ES agents tend to survive by driving slowly or crashing late, showing a clear <strong>sample efficiency gap</strong> compared to PPO.
                            </p>
                        </div>
                    </div>

                    <div className="bg-green-500/10 border border-green-500/20 rounded-md p-3 flex items-start gap-3 mt-2">
                        <div className="text-xs text-green-200/80 font-mono">
                            <strong>UPDATE:</strong> "Safety Governor" removed. Both agents are now operating fully autonomously. RL has achieved <span className="text-green-400">100% completion rate</span> on the Figure-8 track.
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
};
