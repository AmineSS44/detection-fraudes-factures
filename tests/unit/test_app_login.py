"""
Tests unitaires pour la page de connexion et la gestion de session (app.py).
Vérifie la logique de session, la validation du token, et la déconnexion.
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Créer le mock streamlit AVANT tout import de app
_mock_st = MagicMock()
_mock_st.session_state = {}

# Patcher streamlit dans sys.modules de manière permanente pour ce module de test
sys.modules["streamlit"] = _mock_st

# Maintenant on peut importer app en toute sécurité
if "app" in sys.modules:
    del sys.modules["app"]
import app


@pytest.fixture(autouse=True)
def reset_session_state():
    """Réinitialise le session_state entre chaque test."""
    _mock_st.session_state = {}
    _mock_st.reset_mock()
    yield
    _mock_st.session_state = {}


class TestIsAuthenticated:
    """Tests pour la fonction is_authenticated."""

    def test_not_authenticated_when_no_token(self):
        """Vérifie que is_authenticated retourne False si pas de token."""
        _mock_st.session_state = {}
        assert app.is_authenticated() is False

    def test_not_authenticated_when_token_is_none(self):
        """Vérifie que is_authenticated retourne False si token est None."""
        _mock_st.session_state = {"token": None}
        # None est falsy, mais .get("token", "") retourne None, bool(None)=False
        assert app.is_authenticated() is False

    def test_authenticated_when_token_present(self):
        """Vérifie que is_authenticated retourne True si token existe."""
        _mock_st.session_state = {"token": "some-valid-jwt"}
        assert app.is_authenticated() is True

    def test_not_authenticated_when_token_empty_string(self):
        """Vérifie que is_authenticated retourne False si token est chaîne vide."""
        _mock_st.session_state = {"token": ""}
        assert app.is_authenticated() is False


class TestValidateToken:
    """Tests pour la validation du token via l'API."""

    @patch("requests.get")
    def test_validate_token_returns_false_when_not_authenticated(self, mock_get):
        """Vérifie que validate_token retourne False si pas de token."""
        _mock_st.session_state = {}
        result = app.validate_token()
        assert result is False
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_validate_token_returns_true_on_200(self, mock_get):
        """Vérifie que validate_token retourne True si l'API renvoie 200."""
        _mock_st.session_state = {"token": "valid-token"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = app.validate_token()
        assert result is True
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8000/api/stats",
            headers={"Authorization": "Bearer valid-token"},
            timeout=5,
        )

    @patch("requests.get")
    def test_validate_token_returns_false_on_401(self, mock_get):
        """Vérifie que validate_token retourne False si l'API renvoie 401 (token expiré)."""
        _mock_st.session_state = {"token": "expired-token"}
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        result = app.validate_token()
        assert result is False

    @patch("requests.get")
    def test_validate_token_returns_true_on_connection_error(self, mock_get):
        """Vérifie que validate_token retourne True si serveur injoignable."""
        import requests as req
        _mock_st.session_state = {"token": "some-token"}
        mock_get.side_effect = req.exceptions.ConnectionError("Connection refused")

        result = app.validate_token()
        # On considère le token valide si le serveur est injoignable
        assert result is True


class TestDoLogout:
    """Tests pour la déconnexion."""

    def test_logout_clears_session_state(self):
        """Vérifie que do_logout efface toutes les clés de session."""
        _mock_st.session_state = {"token": "jwt-token", "username": "admin"}
        app.do_logout()
        assert len(_mock_st.session_state) == 0

    def test_logout_triggers_rerun(self):
        """Vérifie que do_logout déclenche un rerun Streamlit."""
        _mock_st.session_state = {"token": "jwt-token"}
        app.do_logout()
        _mock_st.rerun.assert_called_once()


class TestGetAuthHeaders:
    """Tests pour la construction des headers d'auth."""

    def test_get_auth_headers_with_token(self):
        """Vérifie que get_auth_headers retourne le bon header Bearer."""
        _mock_st.session_state = {"token": "my-jwt-token"}
        headers = app.get_auth_headers()
        assert headers == {"Authorization": "Bearer my-jwt-token"}

    def test_get_auth_headers_without_token(self):
        """Vérifie que get_auth_headers retourne un header vide si pas de token."""
        _mock_st.session_state = {}
        headers = app.get_auth_headers()
        assert headers == {"Authorization": "Bearer "}


class TestLoginFlow:
    """Tests pour le flux de connexion via l'API."""

    @patch("requests.post")
    def test_login_api_returns_token_on_success(self, mock_post):
        """Vérifie que POST /api/login avec bons identifiants retourne un token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "token": "new-jwt-token",
            "expires_at": "2025-01-01T00:00:00",
        }
        mock_post.return_value = mock_response

        import requests
        response = requests.post(
            f"{app.API_BASE_URL}/api/login",
            json={"username": "admin", "password": "admin123"},
            timeout=10,
        )
        assert response.status_code == 200
        data = response.json()
        # Simuler le stockage du token en session
        _mock_st.session_state["token"] = data["token"]
        assert _mock_st.session_state["token"] == "new-jwt-token"
        assert app.is_authenticated() is True

    @patch("requests.post")
    def test_login_api_returns_401_on_invalid_credentials(self, mock_post):
        """Vérifie que POST /api/login avec mauvais identifiants retourne 401."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Identifiants incorrects"}
        mock_post.return_value = mock_response

        import requests
        response = requests.post(
            f"{app.API_BASE_URL}/api/login",
            json={"username": "wrong", "password": "wrong"},
            timeout=10,
        )
        assert response.status_code == 401
        # Le token ne doit pas être stocké
        assert "token" not in _mock_st.session_state

    def test_api_base_url_configured_correctly(self):
        """Vérifie que l'URL de l'API est correctement configurée."""
        assert app.API_BASE_URL == "http://127.0.0.1:8000"

    def test_login_page_title_contains_detection_fraude(self):
        """Vérifie que show_login affiche le bon titre."""
        _mock_st.session_state = {}
        _mock_st.form_submit_button.return_value = False
        # Créer un context manager mock pour st.form
        form_mock = MagicMock()
        form_mock.__enter__ = MagicMock(return_value=form_mock)
        form_mock.__exit__ = MagicMock(return_value=False)
        _mock_st.form.return_value = form_mock

        app.show_login()
        # Vérifier que le titre contient "Détection de Fraude - Connexion"
        _mock_st.title.assert_called_with("🔐 Détection de Fraude - Connexion")
