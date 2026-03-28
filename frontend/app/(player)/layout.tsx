"use client";

import { ChartBar, Home, type LucideIcon, Video, User } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { useEffect } from "react";

import { useSession } from "@/hooks/use-session";
import { cn } from "@/lib/utils";

const tabs = [
  { label: "Home", icon: Home, href: "/home" },
  { label: "Session", icon: Video, href: "/session/solo" },
  { label: "Stats", icon: ChartBar, href: "/history" },
  { label: "Profile", icon: User, href: "/profile" },
];

function TabItem({
  tab,
  isActive,
  onClick,
}: {
  tab: { label: string; icon: LucideIcon; href: string };
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex flex-1 flex-col items-center justify-center gap-1 rounded-[26px] py-2 transition-all",
        isActive
          ? "bg-linear-to-r from-gradient-start to-gradient-end"
          : ""
      )}
    >
      <tab.icon
        className={cn(
          "h-[18px] w-[18px]",
          isActive ? "text-white" : "text-text-muted"
        )}
      />
      <span
        className={cn(
          "text-[10px] uppercase tracking-[0.5px]",
          isActive ? "font-semibold text-white" : "font-medium text-text-muted"
        )}
      >
        {tab.label}
      </span>
    </button>
  );
}

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
    <div className="flex min-h-dvh w-full flex-col overflow-x-hidden bg-bg-primary">
      <main className="flex-1 overflow-y-auto pb-28">{children}</main>

      <nav className="fixed bottom-0 left-0 z-50 w-full pt-3 px-[21px] pb-[max(21px,env(safe-area-inset-bottom))]">
        <div className="flex h-[62px] items-center rounded-[36px] border border-border bg-bg-card/70 p-1 backdrop-blur-xl">
          {tabs.map((tab) => {
            const isActive =
              pathname === tab.href ||
              (tab.href === "/session/solo" && pathname.startsWith("/session")) ||
              (tab.href === "/home" && pathname === "/home");

            return (
              <TabItem
                key={tab.href}
                tab={tab}
                isActive={isActive}
                onClick={() => router.push(tab.href)}
              />
            );
          })}
        </div>
      </nav>
    </div>
  );
}
