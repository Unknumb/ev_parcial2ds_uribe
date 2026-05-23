"""
Módulo de Preprocesamiento de Datos.
Contiene las funciones para limpiar y unificar los datos crudos, preparándolos para modelamiento.
"""
import pandas as pd

def limpiar_estudiantes(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia textos y estandariza el dataset de estudiantes."""
    df_limpio = df.copy()
    
    # Limpiar textos de espacios invisibles y dejar todo en formato Título o Mayúscula
    df_limpio['estado_matricula'] = df_limpio['estado_matricula'].str.strip().str.upper()
    df_limpio['carrera'] = df_limpio['carrera'].str.strip().str.title()
    df_limpio['sede'] = df_limpio['sede'].str.strip().str.title()
    
    # Eliminar duplicados exactos si los hay
    df_limpio = df_limpio.drop_duplicates()
    
    return df_limpio

def limpiar_calificaciones(df: pd.DataFrame) -> pd.DataFrame:
    """Arregla formatos numéricos y elimina notas atípicas."""
    df_limpio = df.copy()
    
    # 1. Forzar notas a números (convierte textos raros a NaN)
    df_limpio['nota'] = pd.to_numeric(df_limpio['nota'], errors='coerce')
    
    # 2. Eliminar las filas donde la nota se volvió nula por ser un texto irreconocible
    df_limpio = df_limpio.dropna(subset=['nota'])
    
    # 3. Eliminar los Outliers (Notas menores a 1.0 o mayores a 7.0 como el famoso 82.1)
    df_limpio = df_limpio[(df_limpio['nota'] >= 1.0) & (df_limpio['nota'] <= 7.0)]
    
    return df_limpio

def limpiar_asistencia(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza los estados de asistencia."""
    df_limpio = df.copy()
    df_limpio['estado_asistencia'] = df_limpio['estado_asistencia'].str.strip().str.upper()
    return df_limpio

def limpiar_inscripciones(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica de inscripciones."""
    df_limpio = df.copy()
    df_limpio = df_limpio.drop_duplicates()
    return df_limpio

def crear_tabla_maestra(df_estudiantes: pd.DataFrame, df_inscripciones: pd.DataFrame, 
                        df_calificaciones: pd.DataFrame, df_asistencia: pd.DataFrame) -> pd.DataFrame:
    """Une los datasets limpios creando una vista analítica maestra."""
    
    # 1. Resumir Asistencia (Usando la lógica que descubrimos en el EDA)
    df_asistencia['es_ausente'] = (df_asistencia['estado_asistencia'] == 'AUSENTE').astype(int)
    resumen_asistencia = df_asistencia.groupby('id_inscripcion')['es_ausente'].sum().reset_index(name='total_ausencias')
    
    # 2. Resumir Calificaciones (Promedio por inscripción)
    resumen_notas = df_calificaciones.groupby('id_inscripcion')['nota'].mean().reset_index(name='promedio_notas')
    
    # 3. Los JOINS (El cruce de la información)
    # Unimos estudiantes con sus inscripciones
    df_master = df_estudiantes.merge(df_inscripciones, on='id_estudiante', how='inner')
    
    # Le pegamos el resumen de ausencias (how='left' por si un alumno aún no tiene clases registradas)
    df_master = df_master.merge(resumen_asistencia, on='id_inscripcion', how='left')
    
    # Le pegamos el promedio de notas
    df_master = df_master.merge(resumen_notas, on='id_inscripcion', how='left')
    
    # 4. Limpieza final de la tabla maestra
    # Si alguien no tenía asistencia registrada, su total de ausencias es 0
    df_master['total_ausencias'] = df_master['total_ausencias'].fillna(0)
    
    return df_master