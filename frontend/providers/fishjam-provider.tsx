"use client";

import type { ReactNode } from "react";
import { createContext, useContext } from "react";

interface FishjamContextValue {
  peers: Map<string, MediaStream | null>;
  isConnected: boolean;
}

const FishjamContext = createContext<FishjamContextValue>({
  peers: new Map(),
  isConnected: false,
});

export function FishjamProvider({ children }: { children: ReactNode }) {
  return (
    <FishjamContext.Provider value={{ peers: new Map(), isConnected: false }}>
      {children}
    </FishjamContext.Provider>
  );
}

export function useFishjamPeers() {
  return useContext(FishjamContext);
}
