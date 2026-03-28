"use client";

import { useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { useEffect } from "react";

import { useSession } from "@/hooks/use-session";

export default function CoachLayout({ children }: { children: ReactNode }) {
  const router = useRouter();
  const { isAuthenticated, role } = useSession();

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace("/");
    } else if (role !== "coach") {
      router.replace("/home");
    }
  }, [isAuthenticated, role, router]);

  if (!isAuthenticated || role !== "coach") return null;

  return (
    <>
      {/* Mobile warning */}
      <div className="flex min-h-dvh items-center justify-center bg-bg-surface p-8 lg:hidden">
        <p className="text-center text-text-secondary">
          Coach dashboard requires a desktop browser.
        </p>
      </div>
      {/* Desktop layout */}
      <div className="hidden min-h-screen bg-bg-surface lg:block">{children}</div>
    </>
  );
}
