

export function WelcomeScreen() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center p-4 text-center animate-in fade-in duration-500">
      <div className="mb-8 flex h-24 w-24 items-center justify-center overflow-hidden rounded-2xl ring-1 ring-border shadow-sm">
        <img 
          src="/logo-compressed.jpg" 
          alt="Team Knowledge Base Logo" 
          className="h-full w-full object-cover"
        />
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
