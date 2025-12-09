from types import SimpleNamespace

import pytest

import datacreek.security.tenant_privacy as tp


class DummyTenantPrivacy:
    def __init__(self, tenant_id, epsilon_max, epsilon_used=0.0):
        self.tenant_id = tenant_id
        self.epsilon_max = epsilon_max
        self.epsilon_used = epsilon_used


class DummySession:
    def __init__(self):
        self.entries = {}
        self.commits = 0

    def get(self, model, tenant_id):
        return self.entries.get(tenant_id)

    def add(self, entry):
        self.entries[entry.tenant_id] = entry

    def commit(self):
        self.commits += 1

    class _Query:
        def __init__(self, entries):
            self._entries = entries

        def all(self):
            return list(self._entries.values())

    def query(self, model):
        return self._Query(self.entries)


def test_set_and_get_budget(monkeypatch):
    monkeypatch.setattr(tp, "TenantPrivacy", DummyTenantPrivacy)
    db = DummySession()
    tp.set_tenant_limit(db, 1, 1.0)
    assert db.entries[1].epsilon_max == 1.0
    tp.set_tenant_limit(db, 1, 2.0)
    assert db.entries[1].epsilon_max == 2.0
    assert db.commits == 2
    info = tp.get_budget(db, 1)
    assert info == {"epsilon_max": 2.0, "epsilon_used": 0.0, "epsilon_remaining": 2.0}


def test_consume_and_reset(monkeypatch):
    monkeypatch.setattr(tp, "TenantPrivacy", DummyTenantPrivacy)
    db = DummySession()
    tp.set_tenant_limit(db, 2, 1.0)
    assert tp.can_consume_epsilon(db, 2, 0.6)
    assert not tp.can_consume_epsilon(db, 2, 0.5)
    info = tp.get_budget(db, 2)
    assert info["epsilon_used"] == pytest.approx(0.6)
    tp.reset_all(db)
    assert db.entries[2].epsilon_used == 0.0
    info2 = tp.get_budget(db, 2)
    assert info2["epsilon_remaining"] == pytest.approx(1.0)
    assert tp.can_consume_epsilon(db, 99, 0.1) is False
