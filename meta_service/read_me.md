# Documentación API: Modelo de Riesgo (Débito Automático)

Este servicio expone el modelo de Machine Learning para la evaluación de riesgo de fraude en solicitudes de crédito. Recibe un `credit_uid`, procesa la data asociada (SMS, Dispositivos, Metadata) y retorna una decisión de aprobación o rechazo en tiempo real.

---

## ⚙️ Especificaciones Técnicas

| Característica | Detalle |
| :--- | :--- |
| **URL Base** | `https://8meypgbwf5.us-east-1.awsapprunner.com` |
| **Endpoint** | `/run_debito_check/{credit_uid}` |
| **Método** | `POST` |
| **Autenticación** | Pública (Interna) |
| **Formato** | `application/json` |

---

## 📡 Petición (Request)

El endpoint requiere el **ID del crédito** (`credit_uid`) como parámetro en la URL. No requiere cuerpo (body) en la petición.
---
### Estructura del Endpoint
```
https://8meypgbwf5.us-east-1.awsapprunner.com/run_debito_check/{credit_uid}
```
### Ejemplo CUR
```
curl -X 'POST' \
  '[https://8meypgbwf5.us-east-1.awsapprunner.com/run_debito_check/qkRo8MN0pLY6rDwwBSBB](https://8meypgbwf5.us-east-1.awsapprunner.com/run_debito_check/qkRo8MN0pLY6rDwwBSBB)' \
  -H 'accept: application/json' \
  -d ''
```
### Ejemplo de Respuesta Exitosa (`200 OK`)
```
{
  "credit_uid": "qkRo8MN0pLY6rDwwBSBB",
  "user_id": "sWQUWYTXweh2LVfCTmmdfKWNtQc2",
  "device_count": 1,
  "sms_count": 2809,
  "fraud_probability": 0.48595890402793884,
  "decision": "aprobado",
  "threshold": 0.509,
  "features_used": {
    "message_count": 2809,
    "nivel_transaccional": 675,
    "engagement_score": 846,
    "score_riesgo": -220,
    "R_tarjetas_credito_count_vs_ratio_aprobaciones_rechazos": 0.6129,
    "R_credito_formal_vs_tarjetas_credito_count": 1.0526,
    "R_casas_empeno_count_vs_credito_formal": 1.3999,
    "R_vulnerabilidad_financiera_vs_credito_formal": 1.7499,
    "R_educacion_count_vs_credito_formal": 0.2999
    // ... (Variables restantes truncadas para brevedad)
  }
}
```

### Diccionario respuesta

# Diccionario de Datos (Response Fields)

A continuación se detalla el significado de cada campo retornado en el objeto JSON de respuesta.

## 🟢 Campos Principales

| Campo | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `credit_uid` | `String` | **Identificador del Crédito**. Es el ID único que se envió en la petición para realizar el análisis. |
| `user_id` | `String` | **Identificador del Usuario**. ID único del cliente asociado a la solicitud de crédito (obtenido de Firestore). |
| `decision` | `String` | **Decisión Final**. Resultado categórico del modelo.<br>• `aprobado`: El riesgo es bajo (Score < Umbral).<br>• `rechazado`: El riesgo es alto (Score ≥ Umbral). |
| `fraud_probability` | `Float` | **Score de Riesgo (0.0 - 1.0)**. Probabilidad calculada de que el usuario incurra en fraude o impago. A mayor valor, mayor riesgo. |
| `threshold` | `Float` | **Umbral de Corte**. Valor configurado en el modelo para tomar la decisión. Actualmente fijado en **0.509**. |


---
