import pandas as pd
import tensorflow as tf
class DataGeneratorMatch:
    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size

    def crear_dataset(self, df_positivo: pd.DataFrame, df_maestro: pd.DataFrame) -> tf.data.Dataset:
        """
        Genera pares positivos (reales) y negativos sintéticos.
        df_positivo: DataFrame con columnas ['fact_text', 'master_text', 'fact_unit', 'master_unit', 'fact_cost', 'master_cost']
        df_maestro: Catálogo completo de productos para generar negativos aleatorios.
        """
        
        def generator():
            # 1. Preparar datos positivos
            for _, row in df_positivo.iterrows():
                # Par Positivo (Label 1)
                yield ({
                    "fact_text": str(row['fact_text']),
                    "master_text": str(row['master_text']),
                    "fact_unit": str(row['fact_unit']),
                    "master_unit": str(row['master_unit']),
                    "fact_cost": float(row['fact_cost']),
                    "master_cost": float(row['master_cost'])
                }, 1.0)

                # 2. Generar Par Negativo Sintético (Label 0)
                # Tomamos el texto de la factura pero lo cruzamos con un producto aleatorio del maestro
                neg_row = df_maestro.sample(n=1).iloc[0]
                yield ({
                    "fact_text": str(row['fact_text']),
                    "master_text": str(neg_row['master_text']),
                    "fact_unit": str(row['fact_unit']),
                    "master_unit": str(neg_row['master_unit']),
                    "fact_cost": float(row['fact_cost']),
                    "master_cost": float(neg_row['master_cost'])
                }, 0.0)

        # Firma de los datos para TensorFlow
        output_signature = (
            {
                "fact_text": tf.TensorSpec(shape=(), dtype=tf.string),
                "master_text": tf.TensorSpec(shape=(), dtype=tf.string),
                "fact_unit": tf.TensorSpec(shape=(), dtype=tf.string),
                "master_unit": tf.TensorSpec(shape=(), dtype=tf.string),
                "fact_cost": tf.TensorSpec(shape=(), dtype=tf.float32),
                "master_cost": tf.TensorSpec(shape=(), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )

        ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return ds.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)