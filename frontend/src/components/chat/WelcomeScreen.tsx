import { ShieldCheck } from "lucide-react";

export function WelcomeScreen() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center p-4 text-center animate-in fade-in duration-500">
      <div className="mb-8 flex h-20 w-20 items-center justify-center rounded-2xl bg-orange-500/10 ring-1 ring-orange-500/20">
        <ShieldCheck className="h-10 w-10 text-orange-500" />
      </div>
      <h1 className="mb-2 text-2xl font-bold tracking-tight">
        Team Knowledge Base
      </h1>
      <p className="mb-8 text-muted-foreground">
        Access your organization's documentation and insights
      </p>
    </div>
  );
}
