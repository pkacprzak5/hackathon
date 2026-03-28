"use client";

import { ChartBar, Home, Timer, User } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { useEffect } from "react";

import { useSession } from "@/hooks/use-session";
import { cn } from "@/lib/utils";

const tabs = [
  { label: "Home", icon: Home, href: "/home" },
  { label: "Session", icon: Timer, href: "/session/solo" },
  { label: "History", icon: ChartBar, href: "/history" },
  { label: "Profile", icon: User, href: "/profile" },
];

export default function PlayerLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { isAuthenticated } = useSession();

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace("/");
    }
  }, [isAuthenticated, router]);

  if (!isAuthenticated) return null;

  return (
    <div className="flex min-h-dvh w-full flex-col overflow-x-hidden bg-bg-surface">
      <main className="flex-1 overflow-y-auto pb-20">{children}</main>

      <nav className="fixed bottom-0 left-0 z-50 w-full border-t border-border-light bg-bg-card/80 backdrop-blur-xl">
        <div className="flex items-center justify-around py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))]">
          {tabs.map((tab) => {
            const isActive =
              pathname === tab.href ||
              (tab.href === "/session/solo" && pathname.startsWith("/session")) ||
              (tab.href === "/home" && pathname === "/home");

            return (
              <button
                key={tab.href}
                onClick={() => router.push(tab.href)}
                className="flex flex-col items-center gap-1 px-4 py-1"
              >
                <tab.icon
                  className={cn(
                    "h-5 w-5 transition-colors",
                    isActive ? "text-gradient-start" : "text-text-muted"
                  )}
                />
                <span
                  className={cn(
                    "text-[10px] font-medium transition-colors",
                    isActive ? "text-gradient-start" : "text-text-muted"
                  )}
                >
                  {tab.label}
                </span>
                {isActive && (
                  <div className="h-1 w-4 rounded-full bg-linear-to-r from-gradient-start to-gradient-end" />
                )}
              </button>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
