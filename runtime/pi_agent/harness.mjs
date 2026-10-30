/** Isolated official Pi harness. Scheduling, tool validation, transcript and replay are SDK-owned. */
import { mkdirSync } from 'node:fs';
import { createAgentSession, DefaultResourceLoader, SettingsManager } from '@earendil-works/pi-coding-agent';

export async function createIsolatedSession({ cwd, agentDir, modelRuntime, resolved, sessionManager, system, customTools = [], extensionFactory }) {
  mkdirSync(cwd, { recursive: true });
  mkdirSync(agentDir, { recursive: true });
  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: false, keepRecentTokens: 0 }, retry: { enabled: false, maxRetries: 0, provider: { maxRetries: 0 } },
    cacheWarming: 'off', steeringMode: 'all', followUpMode: 'all',
    enableAnalytics: false, enableInstallTelemetry: false, images: { autoResize: false },
  });
  const resourceLoader = new DefaultResourceLoader({
    cwd, agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    extensionFactories: [(pi) => {
      // Pi customPrompt still appends cwd, and an empty customPrompt selects its
      // coding persona. The official before_agent_start override sets
      // forceSystemPrompt, preserving the exact business system even when empty.
      pi.on('before_agent_start', () => ({ systemPrompt: system }));
      return extensionFactory?.(pi);
    }],
    systemPromptOverride: () => system, appendSystemPromptOverride: () => [],
  });
  await resourceLoader.reload();
  const created = await createAgentSession({
    cwd, agentDir, modelRuntime, model: resolved.model, thinkingLevel: resolved.thinkingLevel,
    sessionManager, settingsManager, resourceLoader, noTools: 'builtin',
    customTools, tools: customTools.map((tool) => tool.name),
  });
  if (created.modelFallbackMessage || created.session.model?.id !== resolved.model.id
      || created.session.model?.provider !== resolved.model.provider
      || created.session.thinkingLevel !== resolved.thinkingLevel) {
    created.session.dispose();
    throw Object.assign(new Error('Pi changed the explicitly selected model or thinking level'), { code: 'ROUTE_MISMATCH' });
  }
  created.session.agent.toolExecution = 'sequential';
  if (Number.isFinite(resolved.timeoutMs) && resolved.timeoutMs > 0) {
    // Add only a per-provider-response deadline around the SDK's own stream.
    // Business tool execution is outside this timer; no alternate provider loop
    // or Python inference callback is introduced.
    const officialStream = created.session.agent.streamFunction;
    created.session.agent.streamFunction = (model, context, options = {}) => officialStream(model, context, {
      ...options, timeoutMs: resolved.timeoutMs,
      signal: AbortSignal.any([...(options.signal ? [options.signal] : []), AbortSignal.timeout(Math.ceil(resolved.timeoutMs))]),
    });
  }
  await created.session.bindExtensions({ mode: 'rpc', onError: (error) => { throw new Error(`Pi extension failed: ${error.error}`); } });
  return created.session;
}
