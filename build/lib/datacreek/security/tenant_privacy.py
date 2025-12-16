"""Tenant-level differential privacy budgets."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..db import TenantPrivacy


def set_tenant_limit(db: Session, tenant_id: int, epsilon_max: float) -> None:
    """Create or update the privacy budget for ``tenant_id``."""
    entry = db.get(TenantPrivacy, tenant_id)
    if entry is None:
        entry = TenantPrivacy(tenant_id=tenant_id, epsilon_max=epsilon_max)
        db.add(entry)
    else:
        entry.epsilon_max = epsilon_max
    db.commit()


def can_consume_epsilon(db: Session, tenant_id: int, amount: float) -> bool:
    """Return True and update usage if ``amount`` fits the remaining budget."""
    entry = db.get(TenantPrivacy, tenant_id)
    if entry is None:
        return False
    if entry.epsilon_used + amount > entry.epsilon_max:
        return False
    entry.epsilon_used += amount
    db.commit()
    return True


def reset_all(db: Session) -> None:
    """Reset usage for every tenant budget."""

    for entry in db.query(TenantPrivacy).all():
        entry.epsilon_used = 0.0
    db.commit()


def get_budget(db: Session, tenant_id: int) -> dict | None:
    """Return epsilon budget information for ``tenant_id``.

    Parameters
    ----------
    db:
        Active database session.
    tenant_id:
        Identifier of the tenant/user.

    Returns
    -------
    dict | None
        ``{"epsilon_max": float, "epsilon_used": float, "epsilon_remaining": float}``
        if the tenant exists, otherwise ``None``.
    """

    entry = db.get(TenantPrivacy, tenant_id)
    if entry is None:
        return None
    remaining = entry.epsilon_max - entry.epsilon_used
    return {
        "epsilon_max": float(entry.epsilon_max),
        "epsilon_used": float(entry.epsilon_used),
        "epsilon_remaining": float(remaining),
    }
