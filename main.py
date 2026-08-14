from __future__ import annotations

from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .fallback_pool.controller import FallbackPoolController, FallbackPoolSettings
from .fallback_pool.patcher import RunnerPatch


@register(
    "astrbot_plugin_fallback_pool",
    "zjj1280637679-ship-it",
    "基于失败证据、时间衰减与成功恢复动态调整 AstrBot 回退模型池顺序。",
    "0.1.0",
)
class FallbackPoolPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None) -> None:
        super().__init__(context, config)
        self.config = config or {}
        settings = FallbackPoolSettings.from_mapping(_plain_dict(self.config))
        plugin_name = getattr(self, "name", "astrbot_plugin_fallback_pool")
        data_dir = (
            Path(get_astrbot_data_path())
            / "plugin_data"
            / str(plugin_name or "astrbot_plugin_fallback_pool")
        )
        self.controller = FallbackPoolController(
            data_dir,
            settings,
            logger=self.logger,
        )
        self.patch = RunnerPatch()

    async def initialize(self) -> None:
        if not self.controller.settings.enabled:
            self.controller.set_patch_error("配置中已关闭")
            self.logger.info("智能回退模型池已在配置中关闭。")
            return
        try:
            self.patch.install(self.controller)
            self.controller.set_patch_error("")
            self.logger.info("智能回退模型池已接管 ToolLoopAgentRunner 的候选排序。")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.controller.set_patch_error(message)
            self.logger.error("智能回退模型池加载失败：%s", message, exc_info=True)

    @filter.command_group("fallback_pool")
    def fallback_pool():
        """智能回退模型池管理指令。"""

    @filter.permission_type(filter.PermissionType.ADMIN)
    @fallback_pool.command("status")
    async def fallback_pool_status(self, event: AstrMessageEvent):
        """查看模型信任、降权和临时禁用状态。"""
        yield event.plain_result(self.controller.status_text())

    @filter.permission_type(filter.PermissionType.ADMIN)
    @fallback_pool.command("reset")
    async def fallback_pool_reset(
        self,
        event: AstrMessageEvent,
        target: str = "all",
    ):
        """恢复指定模型或全部模型的原始信任。"""
        count, keys = self.controller.reset(target)
        if count == 0:
            yield event.plain_result(f"没有找到匹配记录：{target}")
            return
        preview = _keys_preview(keys)
        yield event.plain_result(f"已恢复 {count} 个模型记录。{preview}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @fallback_pool.command("disable")
    async def fallback_pool_disable(
        self,
        event: AstrMessageEvent,
        target: str,
        minutes: int = 60,
    ):
        """手动临时禁用模型；分钟数小于等于 0 表示无限期。"""
        duration = None if minutes <= 0 else min(minutes, 7 * 24 * 60)
        count, keys = self.controller.disable(target, minutes=duration)
        if count == 0:
            yield event.plain_result(f"没有找到匹配记录：{target}")
            return
        duration_text = "无限期" if duration is None else f"{duration} 分钟"
        yield event.plain_result(
            f"已将 {count} 个模型禁用 {duration_text}。{_keys_preview(keys)}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @fallback_pool.command("enable")
    async def fallback_pool_enable(
        self,
        event: AstrMessageEvent,
        target: str,
    ):
        """取消模型的手动或额度类临时禁用，但保留已有降权证据。"""
        count, keys = self.controller.enable(target)
        if not keys:
            yield event.plain_result(f"没有找到匹配记录：{target}")
            return
        if count == 0:
            yield event.plain_result(
                f"匹配到 {len(keys)} 个模型，但它们当前没有被禁用。"
            )
            return
        yield event.plain_result(f"已重新启用 {count} 个模型。{_keys_preview(keys)}")

    async def terminate(self) -> None:
        self.controller.settings.enabled = False
        self.patch.uninstall()
        await self.controller.close()
        self.logger.info("智能回退模型池已卸载并保存状态。")


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _keys_preview(keys: list[str], limit: int = 5) -> str:
    if not keys:
        return ""
    shown = "、".join(keys[:limit])
    extra = len(keys) - limit
    if extra > 0:
        shown += f" 等 {len(keys)} 项"
    return f"\n{shown}"
