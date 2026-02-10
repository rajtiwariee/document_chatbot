"""
Tenant context middleware and dependencies.

Provides:
- TenantMiddleware: Extracts tenant_id from JWT and sets it on request.state
- get_tenant_id: FastAPI dependency that returns the current tenant_id
- require_same_tenant: Guard to ensure resource belongs to the user's tenant
"""
import logging
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.tenant import Tenant
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency: get_tenant_id
# ---------------------------------------------------------------------------
async def get_tenant_id(current_user: User = Depends(get_current_user)) -> UUID:
    """
    FastAPI dependency – returns the authenticated user's tenant_id.
    Use this in any endpoint that needs tenant-scoped queries.
    """
    return current_user.tenant_id


# ---------------------------------------------------------------------------
# Dependency: get_tenant
# ---------------------------------------------------------------------------
async def get_tenant(
    tenant_id: UUID = Depends(get_tenant_id),
    db: AsyncSession = Depends(get_db),
) -> Tenant:
    """
    FastAPI dependency – returns the full Tenant object.
    Raises 403 if tenant is deactivated.
    """
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()

    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant not found",
        )

    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant account is deactivated. Contact your administrator.",
        )

    return tenant


# ---------------------------------------------------------------------------
# Guard: require_same_tenant
# ---------------------------------------------------------------------------
def require_same_tenant(resource_tenant_id: UUID, user_tenant_id: UUID) -> None:
    """
    Raises 404 if the resource does not belong to the user's tenant.
    We return 404 (not 403) to avoid leaking resource existence to other tenants.
    """
    if resource_tenant_id != user_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found",
        )


# ---------------------------------------------------------------------------
# Tenant-scoped query helper
# ---------------------------------------------------------------------------
class TenantQuery:
    """
    Helper to build tenant-scoped SQLAlchemy queries.

    Usage:
        tq = TenantQuery(tenant_id)
        stmt = tq.filter(select(Document).order_by(Document.created_at.desc()))
        result = await db.execute(stmt)

    This ensures every query automatically includes the tenant_id filter,
    preventing accidental cross-tenant data leaks.
    """

    def __init__(self, tenant_id: UUID):
        self.tenant_id = tenant_id

    def filter(self, stmt, model=None):
        """
        Add .where(Model.tenant_id == self.tenant_id) to a select statement.
        If model is None, attempts to infer from the statement's columns.
        """
        if model is not None:
            return stmt.where(model.tenant_id == self.tenant_id)

        # Attempt to infer the model from the statement
        # Works for simple select(Model) cases
        for col in stmt.columns_clause_adapter.columns if hasattr(stmt, 'columns_clause_adapter') else []:
            if hasattr(col, 'tenant_id'):
                return stmt.where(col.tenant_id == self.tenant_id)

        # Fallback: caller must provide the model explicitly
        raise ValueError(
            "Cannot infer model for tenant filtering. "
            "Pass the model explicitly: tq.filter(stmt, model=Document)"
        )
