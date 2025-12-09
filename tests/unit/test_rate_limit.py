import types

import datacreek.utils.rate_limit as rl


class DummyRedis:
    def __init__(self):
        self.loaded = 0
        self.calls = []

    def script_load(self, script):
        self.loaded += 1
        return "sha"

    def evalsha(self, sha, numkeys, key, rate, burst, now):
        self.calls.append((sha, numkeys, key, rate, burst, now))
        return 1


class DummyRedisModule:
    class exceptions:
        class NoScriptError(Exception):
            pass


def test_consume_token_local(monkeypatch):
    rl.configure(rate=1, burst=2)
    monkeypatch.setattr(rl, "_get_client", lambda: None)
    rl._SCRIPT_SHA = None
    rl._LOCAL_BUCKETS.clear()
    assert rl.consume_token("t", now=0)
    assert rl.consume_token("t", now=0)
    assert not rl.consume_token("t", now=0)
    # token refills after one second
    assert rl.consume_token("t", now=1)


def test_consume_token_redis(monkeypatch):
    dummy = DummyRedis()
    monkeypatch.setattr(rl, "redis", DummyRedisModule)
    rl.configure(client=dummy, rate=1, burst=1)
    assert rl._SCRIPT_SHA == "sha"
    rl.consume_token("x", now=0)
    assert dummy.calls


def test_consume_token_reload(monkeypatch):
    class ReloadClient(DummyRedis):
        def __init__(self):
            super().__init__()
            self.first = True

        def evalsha(self, *args):
            if self.first:
                self.first = False
                raise DummyRedisModule.exceptions.NoScriptError()
            return super().evalsha(*args)

    client = ReloadClient()
    monkeypatch.setattr(rl, "redis", DummyRedisModule)
    rl.configure(client=client, rate=1, burst=1)
    rl.consume_token("z", now=0)
    assert client.loaded >= 1
    assert client.calls
