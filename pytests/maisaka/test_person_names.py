from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

import pytest

from src.common import person_names as names
from src.common.database.database_model import PersonInfo
from src.common.data_models.message_component_data_model import AtComponent, MessageSequence, TextComponent


@pytest.fixture
def people(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)

    @contextmanager
    def session(**kwargs):
        with Session(engine) as db:
            yield db
            db.commit()

    monkeypatch.setattr(names, "get_db_session", session)
    with session() as db:
        db.add_all([
            PersonInfo(person_id="p1", platform="qq", user_id="u1", person_name="主称呼", user_nickname="QQ名", group_cardname='[{"group_id":"g","group_cardname":"群名片"}]'),
            PersonInfo(person_id="p2", platform="qq", user_id="u2", person_name="", user_nickname="另一个QQ名"),
            PersonInfo(person_id="p3", platform="telegram", user_id="u1", person_name="另一个平台", user_nickname="QQ名"),
        ])
    yield session
    engine.dispose()


def test_names_follow_platform_and_user_id_not_card_or_nickname(people):
    assert names.display_name("qq", "u1", "刚改的QQ名") == "主称呼"
    assert names.display_name("telegram", "u1", "QQ名") == "另一个平台"
    assert names.display_name("qq", "u2") == "另一个QQ名"
    assert names.display_name("qq", "u2", "实时QQ名") == "实时QQ名"
    assert names.display_name("qq", "unknown", "新QQ名") == "新QQ名"
    assert names.display_name("qq", "unknown") == "unknown"


def test_primary_name_change_is_visible_without_cache_or_profile_rebuild(people):
    assert names.display_name("qq", "u1") == "主称呼"
    with people() as db:
        person = db.exec(select(PersonInfo).where(PersonInfo.person_id == "p1")).one()
        person.person_name = "新主称呼"
        db.add(person)
    assert names.display_name("qq", "u1") == "新主称呼"
    cached = {"person_id": "p1", "profile_text": "# 人物画像\n主称呼: 老称呼\n事实保持原样"}
    rendered = names.profile_display_names(cached)
    assert rendered["person_name"] == "新主称呼"
    assert "主称呼: 新主称呼" in rendered["profile_text"]
    assert "事实保持原样" in rendered["profile_text"]
    assert "主称呼: 老称呼" in cached["profile_text"]


def test_projection_changes_labels_but_preserves_source_ids_and_content(people):
    from src.maisaka.context.planner_messages import build_planner_user_prefix_from_session_message
    from src.maisaka.pi_runtime.runner import source_fingerprint

    at = AtComponent(target_user_id="u2", target_user_nickname="另一个QQ名", target_user_cardname="另一个群名片")
    source = SimpleNamespace(
        message_id="m", platform="qq", timestamp=datetime(2026, 11, 4, 12), reply_to=None,
        session_id="g", is_notify=False,
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_id="u1", user_nickname="QQ名", user_cardname="群名片"), group_info=None),
        raw_message=MessageSequence([TextComponent("原文里有群友的名字"), at]),
    )
    before = source_fingerprint(source)
    projected = names.project_message_names(source)
    prefix = build_planner_user_prefix_from_session_message(projected)
    assert 'user="主称呼"' in prefix and 'user_id="u1"' in prefix and 'platform="qq"' in prefix
    assert 'group_card=' not in prefix
    assert projected.raw_message.components[1].target_user_nickname == "另一个QQ名"
    assert projected.raw_message.components[1].target_user_cardname == ""
    assert source.message_info.user_info.user_cardname == "群名片"
    assert at.target_user_cardname == "另一个群名片"
    assert projected.raw_message.components[0] is source.raw_message.components[0]
    assert source_fingerprint(source) == before == source_fingerprint(projected)


def test_monitor_uses_current_name_and_does_not_modify_saved_payload(people):
    payload = {"platform": "qq", "user_id": "u1", "speaker_name": "旧群名片", "content": "原文"}
    shown = names.monitor_display_names(payload)
    assert shown["speaker_name"] == "主称呼"
    assert shown["content"] == payload["content"]
    assert payload["speaker_name"] == "旧群名片"


def test_profile_editor_updates_and_clears_the_authoritative_field(people, monkeypatch):
    from src.webui.routers import person as router

    monkeypatch.setattr(router, "get_db_session", people)
    response = router.update_person("p1", router.PersonUpdateRequest(person_name="  新称呼  "))
    assert response.data.person_name == "新称呼"
    assert names.display_name("qq", "u1") == "新称呼"
    router.update_person("p1", router.PersonUpdateRequest(person_name=""))
    assert names.display_name("qq", "u1") == "QQ名"


def test_automatic_user_sync_preserves_manually_maintained_name(people, monkeypatch):
    from src.person_info import person_info

    monkeypatch.setattr(person_info, "get_db_session", people)
    person = person_info.Person.__new__(person_info.Person)
    for key, value in {"is_known": True, "person_id": "p1", "platform": "qq", "user_id": "u1",
                       "nickname": "刚修改的QQ名", "person_name": "过期的主称呼", "name_reason": "旧理由",
                       "memory_points": [], "group_cardname_list": [], "know_since": 0,
                       "last_know": 0, "know_times": 2}.items():
        setattr(person, key, value)
    person.sync_to_database()
    with people() as db:
        stored = db.exec(select(PersonInfo).where(PersonInfo.person_id == "p1")).one()
        assert stored.user_nickname == "刚修改的QQ名"
        assert stored.person_name == "主称呼"


def test_memory_result_exposes_primary_name_for_id_bound_evidence(people):
    from src.maisaka.memory.result_view import memory_result_preview

    raw = {"hits": [{"type": "paragraph", "content": "历史原文", "metadata": {"person_ids": ["p1"]}}]}
    raw["people"] = names.memory_people(raw)
    assert "人物：主称呼" in memory_result_preview(raw, "r1")
