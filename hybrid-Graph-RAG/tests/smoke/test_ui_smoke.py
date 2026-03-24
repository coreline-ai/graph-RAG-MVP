from bs4 import BeautifulSoup
import pytest


pytestmark = pytest.mark.smoke


def test_search_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, "html.parser")
    assert soup.find("h1").text.strip() == "메시지 검색"


def test_search_page_shows_results(client):
    response = client.get("/?q=배포&message_id=msg-1")
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, "html.parser")
    assert "서버 배포 380차 완료했습니다" in soup.get_text()


def test_insights_page_renders(client):
    response = client.get("/insights")
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, "html.parser")
    assert "인사이트" in soup.get_text()
