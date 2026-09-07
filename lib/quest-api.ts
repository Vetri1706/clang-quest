import type { Bootstrap } from './quest-types';
let token = '';
export async function bootstrap(): Promise<Bootstrap> {
  const response = await fetch('/api/bootstrap', {
    cache: 'no-store',
    signal: AbortSignal.timeout(25000),
  });
  if (!response.ok)
    throw new Error(
      'The local engine is not responding. Start Questline and reconnect.',
    );
  const data = (await response.json()) as Bootstrap;
  if (typeof data.csrf !== 'string' || !Array.isArray(data.challenges))
    throw new Error('The local engine returned an invalid session.');
  token = data.csrf;
  return data;
}
export async function api<T>(
  path: string,
  body?: unknown,
  retry = true,
): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers:
      body === undefined
        ? undefined
        : { 'Content-Type': 'application/json', 'X-Questline-Token': token },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
    signal: AbortSignal.timeout(100000),
  });
  if (response.status === 403 && retry) {
    await bootstrap();
    return api<T>(path, body, false);
  }
  const data = (await response.json()) as { error?: string };
  if (!response.ok)
    throw new Error(
      data.error || 'The local engine could not finish this request.',
    );
  return data as T;
}
