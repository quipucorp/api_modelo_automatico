import os
import boto3
import pandas as pd
import numpy as np
import mlflow.sklearn
import firebase_admin
import logging
import json
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Request
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from typing import List, Dict, Any
from itertools import permutations
from contextlib import asynccontextmanager
from google.cloud import pubsub_v1

# ==========================================
# 0. LOGGING SETUP
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebitoAutomatico")

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Helper para logs estructurados
def log_msg(message, prop=None):
    log_message_debug = {
        "severity": "DEBUG",
        "message": message,
    }
    if prop is not None:
        if isinstance(prop, dict):
            log_message_debug["custom_property"] = json.dumps(prop)
        else:
            log_message_debug["custom_property"] = str(prop)
    print(json.dumps(log_message_debug))

# MLFLOW CONFIG
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "fmoreno")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "password_test")
PRODUCTION_RUN_ID = "322928467e9249c1a9a9d221016b6ff7" 
MODEL_ARTIFACT_PATH = "debito_automatico"

# AWS & PUBSUB CONFIG
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DYNAMO_TABLE_DEVICE = "deviceUser"
DYNAMO_TABLE_SMS = "sms"
PUBSUB_TOPIC_NAME = 'projects/quipumarket-c956f/topics/modelo_ai_debito_automatico'

# ==========================================
# 2. APP LIFESPAN (Startup/Shutdown Logic)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Up: Initializing Clients & Loading Model...")
    
    # --- 1. SETUP AUTH (CRÍTICO: Cargar esto PRIMERO) ---
    # Esto configura la variable de entorno ANTES de que cualquier cliente de Google intente usarse
    if os.path.exists("./creds-2.json"):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./creds-2.json"
        logger.info("🔑 Google Credentials loaded from file")
    else:
        logger.warning("⚠️ creds-2.json not found! Assuming Env Vars are set via Docker/Cloud.")

    # --- 2. SETUP MLFLOW ---
    if not MLFLOW_TRACKING_URI:
        logger.error("❌ CRITICAL: MLFLOW_TRACKING_URI env var is missing!")
        # No lanzamos error aquí para permitir depuración local, pero es grave
    
    # Configuramos MLflow
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"🔌 Connecting to MLflow: {MLFLOW_TRACKING_URI}")

    # --- 3. LOAD MODEL ---
    try:
        model_uri = f"runs:/{PRODUCTION_RUN_ID}/{MODEL_ARTIFACT_PATH}"
        logger.info(f"⬇️ Downloading model from: {model_uri} ...")
        app.state.model = mlflow.sklearn.load_model(model_uri)
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # En producción esto debería detener el app, pero para test local lo dejamos pasar
        app.state.model = None

    # --- 4. INIT DATABASES ---
    initialize_databases(app)

    # --- 5. INIT PUBSUB (CORREGIDO: Ahora se inicia DENTRO de lifespan) ---
    # Al estar aquí, ya 've' la variable GOOGLE_APPLICATION_CREDENTIALS seteada arriba
    try:
        app.state.publisher = pubsub_v1.PublisherClient()
        logger.info("📡 Pub/Sub Client Initialized")
    except Exception as e:
        logger.error(f"❌ Pub/Sub Init Failed: {e}")
        app.state.publisher = None

    yield
    logger.info("🛑 Shutting down...")

# ==========================================
# 3. HELPERS & DB INIT
# ==========================================

def initialize_databases(app):
    # Firestore
    try:
        if not firebase_admin._apps:
            if os.path.exists("./creds-2.json"):
                cred = credentials.Certificate("./creds-2.json")
                firebase_admin.initialize_app(cred)
            else:
                firebase_admin.initialize_app()
        app.state.db = firestore.client()
        logger.info("✅ Firestore connected.")
    except Exception as e:
        logger.error(f"❌ Firestore Error: {e}")
        app.state.db = None

    # DynamoDB
    try:
        app.state.dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
        logger.info("✅ DynamoDB connected.")
    except Exception as e:
        logger.error(f"❌ DynamoDB Error: {e}")
        app.state.dynamodb = None

def publish_df_to_pubsub(data, publisher):
    """Envía el resultado a Google Pub/Sub"""
    if not publisher:
        log_msg("⚠️ Skipping Pub/Sub: Client not initialized.")
        return

    try:
        json_str = json.dumps(data, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        future = publisher.publish(PUBSUB_TOPIC_NAME, json_bytes)
        message_id = future.result()
        log_msg(f'Message {message_id} published to {PUBSUB_TOPIC_NAME}.')
    except Exception as e:
        log_msg(f"❌ Error publishing to Pub/Sub: {e}")

# ==========================================
# 4. APP DEFINITION
# ==========================================
# IMPORTANTE: Aquí NO definimos app.publisher = ...
# Eso causaba el error. Lo hacemos dentro de lifespan.
app = FastAPI(title="Debito Automatico Meta-Service", lifespan=lifespan)

# ==========================================
# 5. CONSTANTS & FEATURES
# ==========================================
BASE_FEATURES_LIST = [
    'tarjetas_credito_count', 'zona_premium_count', 'engagement_score', 
    'impuestos_obligaciones_count', 'credito_formal', 'servicios_publicos_count', 
    'prepago_count', 'casas_empeno_count', 'pospago_vs_prepago', 
    'emergencia_medica_count', 'mensajes_positivos_count', 'vulnerabilidad_financiera', 
    'educacion_count', 'pensiones_cesantias_count', 'ratio_aprobaciones_rechazos'
]

SELECTED_FEATURES = [
    'message_count', 'nivel_transaccional', 'engagement_score', 'score_riesgo',       
    'R_tarjetas_credito_count_vs_ratio_aprobaciones_rechazos',       
    'R_zona_premium_count_vs_servicios_publicos_count',       
    'R_zona_premium_count_vs_mensajes_positivos_count',       
    'R_zona_premium_count_vs_vulnerabilidad_financiera',       
    'R_engagement_score_vs_impuestos_obligaciones_count',       
    'R_credito_formal_vs_tarjetas_credito_count',       
    'R_credito_formal_vs_ratio_aprobaciones_rechazos',       
    'R_servicios_publicos_count_vs_tarjetas_credito_count',       
    'R_casas_empeno_count_vs_credito_formal',       
    'R_emergencia_medica_count_vs_mensajes_positivos_count',       
    'R_emergencia_medica_count_vs_ratio_aprobaciones_rechazos',       
    'R_vulnerabilidad_financiera_vs_credito_formal',       
    'R_educacion_count_vs_credito_formal',       
    'R_educacion_count_vs_mensajes_positivos_count',       
    'R_educacion_count_vs_pensiones_cesantias_count',       
    'R_ratio_aprobaciones_rechazos_vs_credito_formal'
]

# ==========================================
# 6. SERVICE CLASSES
# ==========================================
class DeviceUserService:
    def __init__(self, dynamodb_resource):
        self.table = dynamodb_resource.Table(DYNAMO_TABLE_DEVICE) if dynamodb_resource else None
    
    def get_uuid_devices_by_user_id(self, user_id: str) -> List[str]:
        if not self.table: return []
        try:
            uuid_devices = []
            try:
                response = self.table.query(KeyConditionExpression=Key('userId').eq(user_id))
                items = response.get('Items', [])
            except ClientError:
                response = self.table.scan(FilterExpression=Attr('userId').eq(user_id), Limit=50)
                items = response.get('Items', [])
            
            for item in items:
                dev = item.get('uuidDevice')
                if dev and dev not in uuid_devices:
                    uuid_devices.append(dev)
            return uuid_devices
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return []

class SMSService:
    def __init__(self, dynamodb_resource):
        self.table = dynamodb_resource.Table(DYNAMO_TABLE_SMS) if dynamodb_resource else None
    
    def get_sms_by_uuid_devices(self, uuid_devices: List[str]) -> List[dict]:
        if not self.table or not uuid_devices: return []
        all_sms = []
        for uuid_device in uuid_devices:
            try:
                params = {'KeyConditionExpression': Key('uuidDevice').eq(uuid_device), 'ScanIndexForward': False}
                response = self.table.query(**params)
                items = response.get('Items', [])
                all_sms.extend(items)
                if len(all_sms) > 5000: break 
            except Exception:
                continue
        return all_sms

# ==========================================
# 7. FEATURE EXTRACTION (LOGIC)
# ==========================================
class SMSVariableExtractor:
    def __init__(self, sms_data, user_id):
        self.df = self._convert_to_dataframe(sms_data)
        self.user_id = user_id
        
    def _convert_to_dataframe(self, sms_data):
        if not sms_data: return pd.DataFrame()
        df = pd.DataFrame(sms_data)
        if 'body' not in df.columns: df['body'] = df.get('content', df.get('message', ''))
        if 'address' not in df.columns: df['address'] = df.get('sender', df.get('from', 'unknown'))
        return df.fillna('')
    
    def extract_all_variables(self):
        if len(self.df) == 0: return self._get_empty_variables()
        variables = {}
        df = self.df
        
        # --- BASIC COUNTS ---
        variables['message_count'] = len(df)
        variables['nequi_count'] = df['body'].str.contains('nequi', case=False, na=False).sum()
        variables['bancolombia_count'] = df['body'].str.contains('bancolombia', case=False, na=False).sum()
        variables['davivienda_count'] = df['body'].str.contains('davivienda', case=False, na=False).sum()
        variables['cotizante_count'] = df['body'].str.contains('cotizante', case=False, na=False).sum()
        variables['impuesto_count'] = df['body'].str.contains('dian|impuesto', case=False, na=False).sum()
        variables['prepago_count'] = df['body'].str.contains('recarga|prepago', case=False, na=False).sum()
        variables['nomina_count'] = df['body'].str.contains('nomina|proveedor|salario', case=False, na=False).sum()
        variables['educacion_count'] = df['body'].str.contains('escolar|matricula|universidad|colegio', case=False, na=False).sum()
        variables['servicios_publicos_count'] = df['body'].str.contains('epm|codensa|gas natural|acueducto', case=False, na=False).sum()
        variables['salud_seguros_count'] = df['body'].str.contains('eps|sura|sanitas|compensar', case=False, na=False).sum()
        variables['ingresos_count'] = df['body'].str.contains('consignacion|deposito|transferencia recibida', case=False, na=False).sum()
        variables['gastos_count'] = df['body'].str.contains('compra|retiro|pago realizado|debito', case=False, na=False).sum()
        variables['saldo_bajo_count'] = df['body'].str.contains('saldo insuficiente|sin saldo', case=False, na=False).sum()
        variables['cobranza_count'] = df['body'].str.contains('cobranza|cobro|juridico|multa', case=False, na=False).sum()
        variables['mora_count'] = df['body'].str.contains('vencido|mora|atrasado|pendiente', case=False, na=False).sum()
        variables['aprobaciones_count'] = df['body'].str.contains('preaprobado|aprobado|cupo disponible', case=False, na=False).sum()
        variables['rechazos_count'] = df['body'].str.contains('rechazado|negado|no aprobado', case=False, na=False).sum()
        variables['montos_count'] = df['body'].str.contains(r'\$[\d.,]+|\b\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*(?:pesos|cop)', case=False, na=False).sum()
        variables['otp_count'] = df['body'].str.contains(r'\b(?:codigo|clave|otp|token)\b', case=False, na=False).sum()
        variables['transporte_count'] = df['body'].str.contains('transmilenio|sitp|taxi|beat|uber', case=False, na=False).sum()
        variables['inversiones_count'] = df['body'].str.contains('cdt|inversion|rentabilidad', case=False, na=False).sum()
        variables['credito_grande_count'] = df['body'].str.contains('hipoteca|credito vehiculo|leasing', case=False, na=False).sum()
        variables['pensiones_cesantias_count'] = df['body'].str.contains('colpensiones|porvenir|proteccion', case=False, na=False).sum()
        variables['arriendos_count'] = df['body'].str.contains('arriendo|arrendamiento|canon', case=False, na=False).sum()
        variables['cooperativas_count'] = df['body'].str.contains('cooperativa|coomeva|copetran', case=False, na=False).sum()
        variables['casas_empeno_count'] = df['body'].str.contains('prenda|empeño|gota a gota', case=False, na=False).sum()
        variables['corresponsales_count'] = df['body'].str.contains('baloto|via baloto|corresponsal', case=False, na=False).sum()
        variables['subsidios_count'] = df['body'].str.contains('familias en accion|jovenes en accion|subsidio', case=False, na=False).sum()
        variables['tarjetas_credito_count'] = df['body'].str.contains('visa|mastercard|american express', case=False, na=False).sum()
        variables['ahorro_inversion_count'] = df['body'].str.contains('ahorro programado|cdt virtual', case=False, na=False).sum()
        variables['pago_puntual_count'] = df['body'].str.contains('pago exitoso|pago confirmado', case=False, na=False).sum()
        variables['fidelizacion_count'] = df['body'].str.contains('felicitaciones|premio|beneficio', case=False, na=False).sum()
        variables['microcreditos_count'] = df['body'].str.contains('microcredito|credito de consumo', case=False, na=False).sum()
        variables['compras_credito_count'] = df['body'].str.contains('cuotas sin interes|financiacion', case=False, na=False).sum()
        variables['mensajeria_count'] = df['body'].str.contains('domicilio|mensajeria|envio', case=False, na=False).sum()
        variables['gasolina_peajes_count'] = df['body'].str.contains('gasolina|estacion de servicio|peaje', case=False, na=False).sum()
        variables['medicina_prepagada_count'] = df['body'].str.contains('colsanitas|medisanitas|colmedica', case=False, na=False).sum()
        variables['gimnasios_count'] = df['body'].str.contains('gimnasio|bodytech|smartfit', case=False, na=False).sum()
        variables['viajes_count'] = df['body'].str.contains('avianca|latam|viva air|despegar', case=False, na=False).sum()
        variables['seguros_varios_count'] = df['body'].str.contains('seguro de vida|soat|seguro todo riesgo', case=False, na=False).sum()
        variables['impuestos_obligaciones_count'] = df['body'].str.contains('predial|vehiculo|renta|retencion', case=False, na=False).sum()
        variables['restaurantes_count'] = df['body'].str.contains('restaurante|cine|teatro', case=False, na=False).sum()
        variables['groserias_count'] = df['body'].str.contains('hp|hijueputa|malparido', case=False, na=False).sum()
        variables['apuestas_count'] = df['body'].str.contains('apuesta|betplay|wplay', case=False, na=False).sum()
        variables['alcohol_count'] = df['body'].str.contains('cerveza|aguardiente|ron', case=False, na=False).sum()
        variables['comida_rapida_count'] = df['body'].str.contains('dominos|papa johns|burger king', case=False, na=False).sum()
        variables['emergencia_medica_count'] = df['body'].str.contains('emergencia|urgencia medica', case=False, na=False).sum()
        variables['economia_informal_count'] = df['body'].str.contains('rebusque|camello|chambita', case=False, na=False).sum()
        variables['efectivo_count'] = df['body'].str.contains('efectivo|cash|plata en mano', case=False, na=False).sum()
        variables['ventas_catalogo_count'] = df['body'].str.contains('yanbal|avon|natura', case=False, na=False).sum()
        variables['familia_count'] = df['body'].str.contains('hijo|hija|bebe|esposa', case=False, na=False).sum()
        variables['deportes_premium_count'] = df['body'].str.contains('golf|tenis|equitacion', case=False, na=False).sum()
        variables['transporte_particular_count'] = df['body'].str.contains('carro propio|vehiculo particular', case=False, na=False).sum()
        variables['educacion_privada_count'] = df['body'].str.contains('colegio privado|universidad privada', case=False, na=False).sum()
        variables['zona_premium_count'] = df['body'].str.contains('chicó|rosales|virrey', case=False, na=False).sum()
        variables['marcas_lujo_count'] = df['body'].str.contains('apple|iphone|samsung galaxy', case=False, na=False).sum()
        variables['prestamos_conocidos_count'] = df['body'].str.contains('prestame|me prestas|favor presta', case=False, na=False).sum()
        variables['busqueda_empleo_count'] = df['body'].str.contains('vacante|empleo|trabajo', case=False, na=False).sum()
        variables['cancelaciones_count'] = df['body'].str.contains('cancelar pedido|devolucion', case=False, na=False).sum()
        variables['mensajes_positivos_count'] = df['body'].str.contains('gracias|agradezco|excelente servicio', case=False, na=False).sum()
        variables['horarios_sospechosos_count'] = df['body'].str.contains(r'\b(0[1-4]:[0-5]\d\s*am|madrugada)\b', case=False, na=False).sum()
        variables['ciudades_principales_count'] = df['body'].str.contains('bogota|medellin|cali', case=False, na=False).sum()
        variables['productos_financieros_col_count'] = df['body'].str.contains('ahorro a la mano|cuenta amiga', case=False, na=False).sum()
        variables['comercios_populares_count'] = df['body'].str.contains('tiendas d1|justo y bueno|ara', case=False, na=False).sum()
        variables['informalidad_laboral_count'] = df['body'].str.contains('vendedor ambulante|reciclador', case=False, na=False).sum()
        variables['negociacion_regateo_count'] = df['body'].str.contains('rebaja|descuento|negociar', case=False, na=False).sum()
        variables['problemas_climaticos_count'] = df['body'].str.contains('inundacion|derrumbe', case=False, na=False).sum()
        variables['actividades_agro_count'] = df['body'].str.contains('cosecha|siembra|ganado', case=False, na=False).sum()
        variables['internet_conectividad_count'] = df['body'].str.contains('wifi|internet|datos moviles', case=False, na=False).sum()
        variables['violencia_inseguridad_count'] = df['body'].str.contains('robo|atraco|inseguridad', case=False, na=False).sum()
        variables['actividades_comunitarias_count'] = df['body'].str.contains('junta de accion|reunion vecinos', case=False, na=False).sum()
        variables['otros_bancos_count'] = df['body'].str.contains('bbva|colpatria|caja social', case=False, na=False).sum()
        variables['billeteras_count'] = df['body'].str.contains('daviplata|movii|dale', case=False, na=False).sum()
        variables['ecommerce_count'] = df['body'].str.contains('mercadolibre|amazon|netflix', case=False, na=False).sum()
        variables['retail_count'] = df['body'].str.contains('exito|carulla|jumbo|d1', case=False, na=False).sum()
        variables['fintech_count'] = df['body'].str.contains('dineroya|rapicredit', case=False, na=False).sum()
        variables['telco_count'] = df['body'].str.contains('claro|movistar|tigo', case=False, na=False).sum()
        variables['geeks_count'] = df['body'].str.contains('uber|rappi|cabify', case=False, na=False).sum()

        variables['pospago_vs_prepago'] = (
            df['body'].str.contains('pospago', case=False, na=False).sum() - 
            df['body'].str.contains('prepago', case=False, na=False).sum()
        )
        variables['address_count'] = df['address'].nunique()
        variables['pishing_count'] = int(df['address'].str.count(r'\d').ge(10).sum())
        variables['email_count'] = int(df['address'].str.contains('@', na=False).sum())
        variables['whatsapp_count'] = int(df['address'].str.contains('whatsapp|wa.me', case=False, na=False).sum())

        # --- DERIVED VARIABLES ---
        variables.update(self._calculate_derived_variables(variables))
        
        # --- RATIOS ---
        count_cols = [col for col in variables.keys() if col.endswith('_count')]
        variables['total_count'] = sum(variables.get(col, 0) for col in count_cols)
        for col in count_cols:
            ratio_col = col.replace('_count', '_ratio')
            variables[ratio_col] = (variables[col] / variables['total_count']) if variables['total_count'] > 0 else 0
                
        return variables
    
    def _calculate_derived_variables(self, variables):
        derived = {}
        derived['ratio_ingresos_gastos'] = variables['ingresos_count'] / (variables['gastos_count'] + 1)
        derived['ratio_aprobaciones_rechazos'] = variables['aprobaciones_count'] / (variables['rechazos_count'] + 1)
        derived['tiene_nomina'] = int(variables['nomina_count'] > 0)
        derived['tiene_servicios_publicos'] = int(variables['servicios_publicos_count'] > 0)
        derived['tiene_salud'] = int(variables['salud_seguros_count'] > 0)
        derived['alertas_riesgo'] = variables['saldo_bajo_count'] + variables['mora_count'] + variables['cobranza_count'] + variables['rechazos_count']
        derived['nivel_transaccional'] = variables['ingresos_count'] + variables['gastos_count'] + variables['montos_count']
        
        derived['diversificacion_financiera'] = (
            int(variables['nequi_count'] > 0) + int(variables['bancolombia_count'] > 0) +
            int(variables['davivienda_count'] > 0) + int(variables['otros_bancos_count'] > 0) +
            int(variables['billeteras_count'] > 0)
        )
        derived['engagement_score'] = (
            variables['otp_count'] * 2 + derived['nivel_transaccional'] + derived['diversificacion_financiera'] * 3
        )
        derived['estabilidad_laboral'] = (variables['nomina_count'] + variables['pensiones_cesantias_count'] + variables['cooperativas_count'])
        derived['obligaciones_fijas'] = (variables['arriendos_count'] + variables['servicios_publicos_count'] + variables['medicina_prepagada_count'] + variables['gimnasios_count'])
        derived['estres_financiero'] = (variables['casas_empeno_count'] + variables['saldo_bajo_count'] + variables['mora_count'] + variables['cobranza_count'] + variables['rechazos_count'])
        derived['buen_comportamiento_pago'] = (variables['pago_puntual_count'] + variables['fidelizacion_count'] + variables['aprobaciones_count'])
        derived['credito_formal'] = (variables['tarjetas_credito_count'] + variables['microcreditos_count'] + variables['credito_grande_count'])
        derived['ratio_credito_formal_informal'] = derived['credito_formal'] / (variables['casas_empeno_count'] + 1)
        derived['gastos_discrecionales'] = (variables['viajes_count'] + variables['restaurantes_count'] + variables['gimnasios_count'] + variables['ecommerce_count'])
        derived['prevision_financiera'] = (variables['seguros_varios_count'] + variables['ahorro_inversion_count'] + variables['pensiones_cesantias_count'] + variables['inversiones_count'])
        derived['cumplimiento_tributario'] = (variables['impuesto_count'] + variables['impuestos_obligaciones_count'])
        derived['inclusion_financiera'] = (derived['diversificacion_financiera'] + int(variables['subsidios_count'] > 0) + int(variables['cooperativas_count'] > 0) + int(variables['ahorro_inversion_count'] > 0))
        derived['ratio_obligaciones_ingresos'] = derived['obligaciones_fijas'] / (variables['ingresos_count'] + variables['nomina_count'] + 1)
        
        derived['score_riesgo'] = (derived['estres_financiero'] * 2 - derived['buen_comportamiento_pago'] * 3 - derived['estabilidad_laboral'] * 2 + variables['casas_empeno_count'] * 5)
        derived['estabilidad_emocional'] = (variables['mensajes_positivos_count'] - variables['groserias_count'] - variables['alcohol_count'] - variables['apuestas_count'])
        derived['riesgo_comportamental'] = (variables['groserias_count'] + variables['apuestas_count'] + variables['alcohol_count'] + variables['horarios_sospechosos_count'] + variables['prestamos_conocidos_count'])
        derived['nivel_socioeconomico'] = (variables['zona_premium_count'] + variables['educacion_privada_count'] + variables['marcas_lujo_count'] + variables['deportes_premium_count'] + variables['transporte_particular_count'] - variables['economia_informal_count'] - variables['subsidios_count'])
        derived['presion_financiera'] = (variables['prestamos_conocidos_count'] + variables['emergencia_medica_count'] + variables['busqueda_empleo_count'] + variables['economia_informal_count'] + variables['cancelaciones_count'])
        derived['formalidad_financiera'] = (variables['pospago_vs_prepago'] + int(variables['transporte_particular_count'] > 0) + int(variables['educacion_privada_count'] > 0) - variables['efectivo_count'] - variables['economia_informal_count'])
        derived['score_confiabilidad'] = (variables['mensajes_positivos_count'] + variables['pago_puntual_count'] + variables['fidelizacion_count'] - variables['groserias_count'] - variables['cancelaciones_count'] - variables['horarios_sospechosos_count'])
        derived['vulnerabilidad_financiera'] = (variables['prestamos_conocidos_count'] + variables['casas_empeno_count'] + variables['emergencia_medica_count'] + variables['busqueda_empleo_count'] + variables['economia_informal_count'] + variables['ventas_catalogo_count'])
        derived['informalidad_total'] = (variables['economia_informal_count'] + variables['informalidad_laboral_count'] + variables['efectivo_count'] + variables['negociacion_regateo_count'] + variables['ventas_catalogo_count'])
        derived['vulnerabilidad_contextual'] = (variables['violencia_inseguridad_count'] + variables['problemas_climaticos_count'] + variables['actividades_comunitarias_count'] + variables['comercios_populares_count'])
        derived['modernizacion_financiera'] = (variables['productos_financieros_col_count'] + variables['internet_conectividad_count'] + variables['otp_count'] + variables['billeteras_count'] - variables['efectivo_count'])
        
        derived['score_riesgo_final'] = (
            derived['score_riesgo'] * 0.3 + derived['riesgo_comportamental'] * 0.2 + derived['vulnerabilidad_financiera'] * 0.15 +
            derived['informalidad_total'] * 0.1 + derived['presion_financiera'] * 0.15 + derived['vulnerabilidad_contextual'] * 0.1 -
            derived['score_confiabilidad'] * 0.25 - derived['modernizacion_financiera'] * 0.15 - derived['nivel_socioeconomico'] * 0.1
        )
        return derived
    
    def _get_empty_variables(self):
        return {k: 0.0 for k in BASE_FEATURES_LIST}

def mensajes(sms_logs: List[Dict], user_id: str) -> pd.DataFrame:
    extractor = SMSVariableExtractor(sms_logs, user_id)
    variables = extractor.extract_all_variables()
    return pd.DataFrame([variables])

# ==========================================
# 8. FEATURE EXPANSION & SELECTION
# ==========================================
def feature_expansion(df_base: pd.DataFrame) -> pd.DataFrame:
    df_processed = df_base.copy()
    new_features = {}
    epsilon = 0.0001
    
    pairs = list(permutations(BASE_FEATURES_LIST, 2))
    
    for col_a, col_b in pairs:
        if col_a in df_processed.columns and col_b in df_processed.columns:
            name = f"R_{col_a}_vs_{col_b}"
            new_features[name] = df_processed[col_a] / (df_processed[col_b] + epsilon)
    
    df_ratios = pd.DataFrame(new_features, index=df_processed.index)
    df_final = pd.concat([df_processed, df_ratios], axis=1)
    df_final = df_final.replace([np.inf, -np.inf], 0).fillna(0)
    return df_final

def feature_selection(df_expanded: pd.DataFrame) -> pd.DataFrame:
    df_final = pd.DataFrame(index=df_expanded.index)
    for col in SELECTED_FEATURES:
        if col in df_expanded.columns:
            df_final[col] = df_expanded[col]
        else:
            df_final[col] = 0.0
    return df_final[SELECTED_FEATURES]

# ==========================================
# 9. ENDPOINTS
# ==========================================

def get_firestore_metadata(credit_id: str, db) -> Dict[str, Any]:
    if not db: raise Exception("Firestore not active")
    doc = db.collection("credits").document(credit_id).get()
    if doc.exists:
        return doc.to_dict()
    query = db.collection("credits").where('uid', '==', credit_id).limit(1)
    results = list(query.stream())
    return results[0].to_dict() if results else None

@app.post("/run_debito_check/{uid}")
def run_debito_check(uid: str, request: Request):
    # Retrieve clients from app.state (Injected at Startup)
    model = request.app.state.model
    db = request.app.state.db
    dynamodb = request.app.state.dynamodb
    
    if not model:
        raise HTTPException(status_code=500, detail="Model is NOT loaded on server")

    device_service = DeviceUserService(dynamodb)
    sms_service = SMSService(dynamodb)

    try:
        # 1. Metadata
        logger.info(f"Fetching Metadata for Credit: {uid}")
        metadata = get_firestore_metadata(uid, db)
        if not metadata: raise HTTPException(status_code=404, detail="Credit ID not found")
        
        user_id = metadata.get('userId')
        if not user_id: raise HTTPException(status_code=404, detail="User ID not found in Credit doc")

        # 2. Devices
        logger.info(f"Fetching Devices for User: {user_id}")
        uuid_devices = device_service.get_uuid_devices_by_user_id(user_id)
        
        # 3. SMS
        sms_logs = sms_service.get_sms_by_uuid_devices(uuid_devices)

        # 4. Extract (Michael)
        df_base = mensajes(sms_logs, user_id)
        
        # 5. Expand (Ratios)
        logger.info("Running Feature Expansion...")
        df_expanded = feature_expansion(df_base)

        # 6. SELECT (Filter to 20 Features)
        logger.info("Running Feature Selection...")
        df_final = feature_selection(df_expanded)
        
        # 7. Predict
        probs = model.predict_proba(df_final)[:, 1]
        score = float(probs[0])
        
        THRESHOLD = 0.5090
        decision = "aprobado" if score < THRESHOLD else "rechazado"

        result = {
            "credit_uid": uid,
            "user_id": user_id,
            "device_count": len(uuid_devices),
            "sms_count": len(sms_logs),
            "fraud_probability": score,
            "decision": decision,
            "threshold": THRESHOLD,
            "features_used": df_final.to_dict(orient='records')[0]
        }

        # 8. PUBLISH TO PUBSUB (Safe Call)
        # Recuperamos el publisher del ESTADO, no de app.publisher
        publisher = request.app.state.publisher
        if publisher:
            log_msg(f"Publishing result to PubSub...")
            publish_df_to_pubsub(result, publisher)
        else:
            log_msg("⚠️ PubSub client not ready, skipping publish.")

        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in run_debito_check: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)