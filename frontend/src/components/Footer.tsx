import { Heart, Github, Linkedin, Globe } from 'lucide-react';

export const Footer = () => {
    return (
        <footer className="relative mt-20 border-t border-muted bg-card/20 backdrop-blur-sm">
            <div className="container mx-auto px-4 py-8">
                <div className="flex flex-col items-center justify-center gap-4">
                    {/* Main text */}
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <span>Made with</span>
                        <Heart className="w-4 h-4 text-red-500 fill-red-500 animate-pulse" />
                        <span>by</span>
                        <span className="font-semibold text-foreground">Keshav Majithia</span>
                    </div>

                    {/* Social Links */}
                    <div className="flex items-center gap-4">
                        <a
                            href="https://github.com/keshavMajithia/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/20 hover:bg-muted/40 transition-colors text-xs text-muted-foreground hover:text-foreground group"
                        >
                            <Github className="w-4 h-4 group-hover:text-primary transition-colors" />
                            <span>GitHub</span>
                        </a>
                        <a
                            href="https://www.linkedin.com/in/keshav-majithia/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/20 hover:bg-muted/40 transition-colors text-xs text-muted-foreground hover:text-foreground group"
                        >
                            <Linkedin className="w-4 h-4 group-hover:text-blue-500 transition-colors" />
                            <span>LinkedIn</span>
                        </a>
                        <a
                            href="http://kay-alpha.vercel.app/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/20 hover:bg-muted/40 transition-colors text-xs text-muted-foreground hover:text-foreground group"
                        >
                            <Globe className="w-4 h-4 group-hover:text-green-500 transition-colors" />
                            <span>Portfolio</span>
                        </a>
                    </div>

                    {/* Project Timeline */}
                    <div className="text-xs text-muted-foreground/70">
                        Built over 5 months • September 2025 - January 2026
                    </div>

                    {/* Copyright */}
                    <div className="text-xs text-muted-foreground/60">
                        © 2026 Keshav Majithia. All rights reserved.
                    </div>
                </div>
            </div>
        </footer>
    );
};
