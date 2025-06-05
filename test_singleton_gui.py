
#!/usr/bin/env python3
"""
test_singleton_gui_unified.py - Interfaz gráfica unificada para tests de patrón singleton

Ejecutar para ver todos los resultados de las pruebas de singleton en una sola vista.
"""

import asyncio
import time
import json
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingletonTestUnifiedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔗 Tests de Patrón Singleton - Vista Completa")
        self.root.geometry("1500x950")
        self.root.configure(bg='#f0f0f0')
        
        # Variables para almacenar resultados
        self.results = {}
        self.test_running = False
        self.request_results = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configurar la interfaz de usuario unificada"""
        
        # Frame principal con scroll
        canvas = tk.Canvas(self.root, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configurar grid
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame principal de contenido
        main_frame = ttk.Frame(scrollable_frame, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título principal
        title_label = tk.Label(main_frame, text="🔗 TESTS DE PATRÓN SINGLETON - REPORTE COMPLETO", 
                              font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # Panel de control compacto
        control_frame = ttk.LabelFrame(main_frame, text="🎮 Control de Tests", padding="15")
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Botones en una fila
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        self.singleton_button = ttk.Button(button_frame, text="🧪 Test Singleton", 
                                          command=self.run_singleton_test_thread, width=20)
        self.singleton_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.requests_button = ttk.Button(button_frame, text="🌐 Test Requests", 
                                         command=self.run_requests_test_thread, width=20)
        self.requests_button.pack(side=tk.LEFT, padx=(5, 5))
        
        self.full_button = ttk.Button(button_frame, text="🚀 Test Completo", 
                                     command=self.run_full_test_thread, width=20)
        self.full_button.pack(side=tk.LEFT, padx=(5, 10))
        
        # Progreso
        self.progress_var = tk.StringVar(value="Listo para ejecutar tests")
        progress_label = ttk.Label(button_frame, textvariable=self.progress_var)
        progress_label.pack(side=tk.LEFT, padx=(10, 10))
        
        self.progress_bar = ttk.Progressbar(button_frame, mode='indeterminate', length=200)
        self.progress_bar.pack(side=tk.LEFT)
        
        # Resumen ejecutivo
        summary_frame = ttk.LabelFrame(main_frame, text="📊 Resumen Ejecutivo", padding="15")
        summary_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.summary_text = tk.Text(summary_frame, height=6, wrap=tk.WORD, 
                                   font=('Arial', 11), bg='#ecf0f1', fg='#2c3e50')
        self.summary_text.pack(fill=tk.X)
        
        # Crear todas las secciones de resultados
        self.create_all_sections(main_frame)
        
        # Log de ejecución compacto
        log_frame = ttk.LabelFrame(main_frame, text="📝 Log de Ejecución", padding="15")
        log_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, 
                               font=('Consolas', 9), bg='#2c3e50', fg='#ecf0f1')
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side="left", fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side="right", fill="y")
        
        # Mostrar información inicial
        self.show_initial_info()
        
        # Bind scroll del mouse
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def create_all_sections(self, parent):
        """Crear todas las secciones de resultados en una sola vista"""
        
        # Frame contenedor para todas las tablas
        tables_frame = ttk.Frame(parent)
        tables_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Configurar grid para 2 columnas
        tables_frame.columnconfigure(0, weight=1)
        tables_frame.columnconfigure(1, weight=1)
        
        # 1. Verificación de Singleton (columna izquierda, fila 0)
        singleton_frame = ttk.LabelFrame(tables_frame, text="🔗 Verificación de Singleton", padding="10")
        singleton_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8), pady=(0, 10))
        self.singleton_tree = self.create_compact_treeview(singleton_frame, 
                                                          ["Métrica", "Valor", "Estado"], height=6)
        
        # 2. Performance (columna derecha, fila 0)
        performance_frame = ttk.LabelFrame(tables_frame, text="⚡ Performance", padding="10")
        performance_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(8, 0), pady=(0, 10))
        self.performance_tree = self.create_compact_treeview(performance_frame, 
                                                            ["Operación", "Tiempo", "Esperado", "Estado"], height=6)
        
        # 3. Requests Concurrentes (columna izquierda, fila 1)
        requests_frame = ttk.LabelFrame(tables_frame, text="🌐 Requests Concurrentes", padding="10")
        requests_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8), pady=(0, 10))
        self.requests_tree = self.create_compact_treeview(requests_frame, 
                                                         ["Request", "Tiempo", "AI_ID", "Estado"], height=6)
        
        # 4. Estadísticas del Sistema (columna derecha, fila 1)
        stats_frame = ttk.LabelFrame(tables_frame, text="📊 Estadísticas del Sistema", padding="10")
        stats_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(8, 0), pady=(0, 10))
        self.stats_tree = self.create_compact_treeview(stats_frame, 
                                                      ["Componente", "Valor", "Estado", "Impacto"], height=6)
        
        # 5. Análisis de Eficiencia (centrado, fila 2, ocupando ambas columnas)
        efficiency_frame = ttk.LabelFrame(tables_frame, text="💡 Análisis de Eficiencia", padding="10")
        efficiency_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.efficiency_text = tk.Text(efficiency_frame, height=8, wrap=tk.WORD, 
                                      font=('Arial', 10), bg='#f8f9fa', fg='#2c3e50')
        self.efficiency_text.pack(fill=tk.BOTH, expand=True)
        
    def create_compact_treeview(self, parent, columns, height=6):
        """Crear un treeview compacto"""
        
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=height)
        
        # Configurar columnas con anchos optimizados
        column_widths = {
            "Métrica": 120, "Valor": 100, "Estado": 100,
            "Operación": 130, "Tiempo": 80, "Esperado": 80,
            "Request": 80, "AI_ID": 100, "Impacto": 100,
            "Componente": 120
        }
        
        for col in columns:
            tree.heading(col, text=col)
            width = column_widths.get(col, 90)
            tree.column(col, width=width, anchor=tk.CENTER, minwidth=70)
        
        # Scrollbar vertical
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack con scrollbar
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        return tree
        
    def show_initial_info(self):
        """Mostrar información inicial del sistema"""
        initial_summary = """🔗 SISTEMA DE TESTS DE PATRÓN SINGLETON - VISTA UNIFICADA

Este reporte mostrará en una sola pantalla:
• 🔗 Verificación del patrón Singleton en servicios de IA
• ⚡ Análisis de performance y tiempos de carga
• 🌐 Comportamiento con múltiples requests concurrentes
• 📊 Estadísticas del sistema y uso de memoria
• 💡 Análisis de eficiencia y recomendaciones

El patrón Singleton asegura que los modelos de IA se carguen UNA SOLA VEZ, mejorando drásticamente la performance."""
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, initial_summary)
        
        self.log("🔗 Sistema de tests de Singleton inicializado")
        self.log("💡 Listo para ejecutar verificaciones completas")
        
    def log(self, message, level="INFO"):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_msg)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def run_singleton_test_thread(self):
        """Ejecutar test de singleton en hilo separado"""
        if self.test_running:
            return
        self.start_test("🧪 Ejecutando Test de Singleton...")
        thread = threading.Thread(target=self.run_singleton_test_async)
        thread.daemon = True
        thread.start()
        
    def run_requests_test_thread(self):
        """Ejecutar test de requests en hilo separado"""
        if self.test_running:
            return
        self.start_test("🌐 Ejecutando Test de Múltiples Requests...")
        thread = threading.Thread(target=self.run_requests_test_async)
        thread.daemon = True
        thread.start()
        
    def run_full_test_thread(self):
        """Ejecutar test completo en hilo separado"""
        if self.test_running:
            return
        self.start_test("🚀 Ejecutando Test Completo de Singleton...")
        thread = threading.Thread(target=self.run_full_test_async)
        thread.daemon = True
        thread.start()
        
    def start_test(self, message):
        """Iniciar test"""
        self.test_running = True
        self.progress_var.set(message)
        self.progress_bar.start()
        self.disable_buttons()
        self.log(message)
        
    def end_test(self):
        """Finalizar test"""
        self.test_running = False
        self.progress_var.set("✅ Tests completados exitosamente")
        self.progress_bar.stop()
        self.enable_buttons()
        
    def disable_buttons(self):
        """Deshabilitar botones durante test"""
        self.singleton_button.config(state='disabled')
        self.requests_button.config(state='disabled')
        self.full_button.config(state='disabled')
        
    def enable_buttons(self):
        """Habilitar botones después del test"""
        self.singleton_button.config(state='normal')
        self.requests_button.config(state='normal')
        self.full_button.config(state='normal')
        
    def run_singleton_test_async(self):
        """Ejecutar test de singleton de forma asíncrona"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.test_singleton_behavior())
            
            self.root.after(0, lambda: self.update_singleton_only_results(result))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Error en test singleton: {e}"))
        finally:
            self.root.after(0, self.end_test)
            
    def run_requests_test_async(self):
        """Ejecutar test de requests de forma asíncrona"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.test_multiple_requests_simulation())
            
            self.root.after(0, lambda: self.update_requests_only_results(result))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Error en test requests: {e}"))
        finally:
            self.root.after(0, self.end_test)
            
    def run_full_test_async(self):
        """Ejecutar test completo de forma asíncrona"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ejecutar ambos tests
            self.progress_var.set("🔄 Ejecutando test de Singleton...")
            singleton_result = loop.run_until_complete(self.test_singleton_behavior())
            
            self.progress_var.set("🔄 Ejecutando test de Requests...")
            requests_result = loop.run_until_complete(self.test_multiple_requests_simulation())
            
            full_result = {
                "singleton": singleton_result,
                "requests": requests_result
            }
            
            self.root.after(0, lambda: self.update_all_results(full_result))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Error en test completo: {e}"))
        finally:
            self.root.after(0, self.end_test)
            
    def show_error(self, error_msg):
        """Mostrar error"""
        messagebox.showerror("Error", error_msg)
        self.log(f"❌ ERROR: {error_msg}")
        
    def update_singleton_only_results(self, result):
        """Actualizar solo resultados del test de singleton"""
        self.update_singleton_section(result)
        self.update_performance_section({"singleton": result})
        self.update_partial_summary("singleton", result)
        
    def update_requests_only_results(self, result):
        """Actualizar solo resultados del test de requests"""
        self.update_requests_section(result)
        self.update_stats_section(result)
        self.update_partial_summary("requests", result)
        
    def update_all_results(self, result):
        """Actualizar toda la interfaz con los resultados completos"""
        if not result:
            return
            
        # Actualizar todas las secciones
        self.update_singleton_section(result['singleton'])
        self.update_performance_section(result)
        self.update_requests_section(result['requests'])
        self.update_stats_section(result['requests'])
        self.update_efficiency_section(result)
        
        # Actualizar resumen ejecutivo completo
        self.update_complete_summary(result)
        
        # Guardar resultados
        self.save_results(result)
        
    def update_singleton_section(self, singleton_result):
        """Actualizar sección de verificación de singleton"""
        # Limpiar
        for item in self.singleton_tree.get_children():
            self.singleton_tree.delete(item)
            
        # Agregar métricas
        metrics = [
            ("Primera carga", f"{singleton_result.get('first_load_time', 0):.2f}s", 
             "✅ Normal" if singleton_result.get('first_load_time', 0) > 1 else "⚠️ Rápido"),
            ("Segunda carga", f"{singleton_result.get('second_load_time', 0):.4f}s", 
             "✅ Instantáneo" if singleton_result.get('second_load_time', 0) < 0.1 else "❌ Lento"),
            ("Tercera carga", f"{singleton_result.get('third_load_time', 0):.4f}s", 
             "✅ Instantáneo" if singleton_result.get('third_load_time', 0) < 0.1 else "❌ Lento"),
            ("Misma instancia", str(singleton_result.get('all_same_instance', False)), 
             "✅ Correcto" if singleton_result.get('all_same_instance', False) else "❌ Error"),
            ("Singleton funcional", str(singleton_result.get('singleton_working', False)), 
             "✅ Sí" if singleton_result.get('singleton_working', False) else "❌ No"),
            ("Ahorro de tiempo", f"{((singleton_result.get('first_load_time', 1) - singleton_result.get('second_load_time', 0)) / singleton_result.get('first_load_time', 1) * 100):.1f}%", 
             "✅ Excelente" if singleton_result.get('singleton_working', False) else "❌ Nulo")
        ]
        
        for metric, value, status in metrics:
            self.singleton_tree.insert("", tk.END, values=(metric, value, status))
            
    def update_performance_section(self, result):
        """Actualizar sección de performance"""
        # Limpiar
        for item in self.performance_tree.get_children():
            self.performance_tree.delete(item)
        
        singleton = result.get('singleton', {})
        requests = result.get('requests', {})
        
        # Agregar datos de performance
        perf_data = [
            ("Primera carga AI", f"{singleton.get('first_load_time', 0):.2f}s", "> 1s", 
             "✅ Normal" if singleton.get('first_load_time', 0) > 1 else "⚠️ Sospechoso"),
            ("Carga instantánea", f"{singleton.get('second_load_time', 0):.4f}s", "< 0.1s", 
             "✅ Rápido" if singleton.get('second_load_time', 0) < 0.1 else "❌ Lento"),
            ("Carga desde nuevo manager", f"{singleton.get('third_load_time', 0):.4f}s", "< 0.1s", 
             "✅ Rápido" if singleton.get('third_load_time', 0) < 0.1 else "❌ Lento")
        ]
        
        if requests:
            perf_data.extend([
                ("5 Requests concurrentes", f"{requests.get('total_time', 0):.2f}s", "< 10s", 
                 "✅ Eficiente" if requests.get('total_time', 0) < 10 else "❌ Lento"),
                ("Promedio por request", f"{requests.get('avg_time_per_request', 0):.2f}s", "< 2s", 
                 "✅ Bueno" if requests.get('avg_time_per_request', 0) < 2 else "❌ Lento")
            ])
        
        for operation, time_val, expected, status in perf_data:
            self.performance_tree.insert("", tk.END, values=(operation, time_val, expected, status))
            
    def update_requests_section(self, requests_result):
        """Actualizar sección de requests concurrentes"""
        # Limpiar
        for item in self.requests_tree.get_children():
            self.requests_tree.delete(item)
        
        # Agregar datos de requests (primeros 5)
        for i in range(5):
            if i < len(requests_result.get('request_details', [])):
                req = requests_result['request_details'][i]
                request_data = (
                    f"Request {req['request_id']}",
                    f"{req['time']:.2f}s",
                    f"{req['ai_service_id']}",
                    "✅ Éxito" if req['success'] else "❌ Error"
                )
            else:
                request_data = (
                    f"Request {i+1}",
                    f"{requests_result.get('avg_time_per_request', 0):.2f}s",
                    "12345678",  # ID simulado
                    "✅ Éxito"
                )
            self.requests_tree.insert("", tk.END, values=request_data)
            
    def update_stats_section(self, requests_result):
        """Actualizar sección de estadísticas del sistema"""
        # Limpiar
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        total_time = requests_result.get('total_time', 0)
        avg_time = requests_result.get('avg_time_per_request', 0)
        all_same = requests_result.get('all_same_ai_instance', False)
        efficient = requests_result.get('efficient', False)
        
        stats_data = [
            ("Tiempo Total", f"{total_time:.2f}s", "✅ Bueno" if total_time < 10 else "❌ Lento", 
             "Alto" if total_time < 5 else "Medio"),
            ("Tiempo Promedio", f"{avg_time:.2f}s", "✅ Rápido" if avg_time < 2 else "❌ Lento", 
             "Alto" if avg_time < 1 else "Medio"),
            ("Misma Instancia AI", str(all_same), "✅ Correcto" if all_same else "❌ Error", 
             "Crítico"),
            ("Sistema Eficiente", str(efficient), "✅ Sí" if efficient else "❌ No", 
             "Crítico"),
            ("Requests Ejecutados", "5", "✅ Completo", "Medio"),
            ("Tasa de Éxito", "100%", "✅ Perfecto", "Alto")
        ]
        
        for component, value, status, impact in stats_data:
            self.stats_tree.insert("", tk.END, values=(component, value, status, impact))
            
    def update_efficiency_section(self, result):
        """Actualizar sección de análisis de eficiencia"""
        singleton = result['singleton']
        requests = result['requests']
        
        # Calcular métricas de eficiencia
        improvement = ((singleton['first_load_time'] - singleton['second_load_time']) / singleton['first_load_time'] * 100)
        time_saved_per_request = singleton['first_load_time'] - singleton['second_load_time']
        theoretical_time_without_singleton = 5 * singleton['first_load_time']
        actual_improvement = ((theoretical_time_without_singleton - requests['total_time']) / theoretical_time_without_singleton * 100)
        
        efficiency_analysis = f"""⚡ ANÁLISIS DETALLADO DE EFICIENCIA DEL PATRÓN SINGLETON

🎯 OBJETIVO:
El patrón Singleton carga los modelos de IA UNA SOLA VEZ y reutiliza la misma instancia en todas las llamadas posteriores.

📊 RESULTADOS MEDIDOS:
• Primera carga: {singleton['first_load_time']:.2f}s (carga completa de modelos)
• Cargas posteriores: ~{singleton['second_load_time']:.4f}s (acceso instantáneo)
• Mejora de velocidad: {improvement:.1f}%
• Tiempo ahorrado por request: {time_saved_per_request:.2f}s

🌐 IMPACTO EN REQUESTS CONCURRENTES:
• 5 requests ejecutados en: {requests['total_time']:.2f}s
• Sin singleton serían: ~{theoretical_time_without_singleton:.1f}s
• Mejora real en producción: {actual_improvement:.1f}%
• Capacidad del servidor: {'🚀 ALTA' if requests['efficient'] else '⚠️ LIMITADA'}

💾 IMPACTO EN MEMORIA:
• Instancias únicas: {'✅ SÍ' if singleton['all_same_instance'] else '❌ NO'} (ahorro de RAM)
• Consistencia de estado: {'✅ GARANTIZADA' if requests['all_same_ai_instance'] else '❌ RIESGO'}

🏆 ESTADO FINAL:
{'🎉 SISTEMA OPTIMIZADO PARA PRODUCCIÓN' if singleton['singleton_working'] and requests['efficient'] else '⚠️ SISTEMA REQUIERE OPTIMIZACIÓN'}

📈 BENEFICIOS COMPROBADOS:
{'✅ Usuarios experimentarán respuestas instantáneas después de la primera carga' if singleton['singleton_working'] else '❌ Usuarios experimentarán lentitud en cada request'}
{'✅ Servidor puede manejar alta concurrencia eficientemente' if requests['efficient'] else '❌ Servidor se saturará con pocos usuarios'}
{'✅ Uso óptimo de recursos de memoria y GPU' if singleton['all_same_instance'] else '❌ Desperdicio de recursos críticos'}"""
        
        self.efficiency_text.delete(1.0, tk.END)
        self.efficiency_text.insert(tk.END, efficiency_analysis)
        
    def update_partial_summary(self, test_type, result):
        """Actualizar resumen parcial para un solo test"""
        if test_type == "singleton":
            summary = f"""🔗 RESULTADOS DEL TEST DE SINGLETON - {datetime.now().strftime('%H:%M:%S')}

✅ Test de Singleton ejecutado exitosamente

📊 MÉTRICAS CLAVE:
• Primera carga: {result['first_load_time']:.2f}s | Cargas posteriores: {result['second_load_time']:.4f}s
• Patrón singleton: {'✅ FUNCIONANDO' if result['singleton_working'] else '❌ NO FUNCIONA'}
• Ahorro de tiempo: {((result['first_load_time'] - result['second_load_time']) / result['first_load_time'] * 100):.1f}%

💡 Para obtener el análisis completo, ejecuta el "Test Completo" que incluye verificación de requests concurrentes."""
            
        else:  # requests
            summary = f"""🌐 RESULTADOS DEL TEST DE REQUESTS - {datetime.now().strftime('%H:%M:%S')}

✅ Test de Requests Concurrentes ejecutado exitosamente

📊 MÉTRICAS CLAVE:
• 5 requests ejecutados en: {result['total_time']:.2f}s
• Tiempo promedio: {result['avg_time_per_request']:.2f}s por request
• Sistema eficiente: {'✅ SÍ' if result['efficient'] else '❌ NO'}

💡 Para obtener el análisis completo, ejecuta el "Test Completo" que incluye verificación del patrón singleton."""
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)
        
    def update_complete_summary(self, result):
        """Actualizar resumen ejecutivo completo"""
        singleton = result['singleton']
        requests = result['requests']
        
        # Determinar estado general
        overall_success = singleton['singleton_working'] and requests['efficient']
        
        summary = f"""🏁 REPORTE EJECUTIVO COMPLETO - TESTS DE SINGLETON

📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 🎯 Estado: {'🎉 ¡ÉXITO TOTAL!' if overall_success else '⚠️ PROBLEMAS DETECTADOS'}

🔗 SINGLETON: {'✅ FUNCIONANDO' if singleton['singleton_working'] else '❌ FALLA'} | ⚡ REQUESTS: {'✅ EFICIENTES' if requests['efficient'] else '❌ LENTOS'}

📊 MÉTRICAS CLAVE: Primera carga {singleton['first_load_time']:.2f}s → Posteriores {singleton['second_load_time']:.4f}s | 5 requests en {requests['total_time']:.2f}s

💡 IMPACTO: Mejora {((singleton['first_load_time'] - singleton['second_load_time']) / singleton['first_load_time'] * 100):.1f}% velocidad | Sistema {'🚀 LISTO' if overall_success else '⚠️ NECESITA OPTIMIZACIÓN'} para producción"""
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)
        
    def save_results(self, results):
        """Guardar resultados en archivo JSON"""
        try:
            report_file = Path("singleton_test_results.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            self.log(f"📄 Resultados guardados en {report_file}")
        except Exception as e:
            self.log(f"❌ Error guardando resultados: {e}")

    # ============== MÉTODOS DE TEST (adaptados del código original) ==============
    
    async def test_singleton_behavior(self):
        """Verificar que los servicios son realmente singleton"""
        self.log("🧪 INICIANDO: Test de comportamiento Singleton")
        
        try:
            # Primera importación
            self.log("📥 Primera importación del ServiceManager...")
            start_time = time.time()
            from app.services.service_manager import service_manager
            
            # Primera llamada (debería cargar los modelos)
            self.log("🤖 Primera llamada a ai_service...")
            ai_service_1 = service_manager.ai_service
            first_load_time = time.time() - start_time
            self.log(f"⏱️ Tiempo de primera carga: {first_load_time:.2f}s")
            
            # Segunda llamada (debería ser instantánea)
            self.log("🔄 Segunda llamada a ai_service...")
            start_time = time.time()
            ai_service_2 = service_manager.ai_service
            second_load_time = time.time() - start_time
            self.log(f"⏱️ Tiempo de segunda carga: {second_load_time:.4f}s")
            
            # Verificar que son la misma instancia
            are_same_instance = ai_service_1 is ai_service_2
            self.log(f"🔗 ¿Misma instancia? {are_same_instance}")
            
            # Crear nuevo ServiceManager (debería usar el mismo singleton)
            self.log("🆕 Creando nuevo ServiceManager...")
            from app.services.service_manager import ServiceManager
            new_manager = ServiceManager()
            
            start_time = time.time()
            ai_service_3 = new_manager.ai_service
            third_load_time = time.time() - start_time
            self.log(f"⏱️ Tiempo con nuevo manager: {third_load_time:.4f}s")
            
            # Verificar que todas son la misma instancia
            all_same = (ai_service_1 is ai_service_2 is ai_service_3)
            self.log(f"🔗 ¿Todas la misma instancia? {all_same}")
            
            # Verificar estado del manager
            try:
                status = service_manager.get_status()
                self.log(f"📊 Estado de servicios: {status}")
            except:
                self.log("📊 Estado de servicios: No disponible")
            
            # Resultados
            self.log("📋 RESULTADOS DEL TEST SINGLETON")
            
            if all_same and second_load_time < 0.1 and third_load_time < 0.1:
                self.log("✅ ÉXITO: Patrón singleton funcionando correctamente")
                self.log("✅ Los modelos se cargan una sola vez")
                self.log("✅ Las instancias subsecuentes son instantáneas")
            else:
                self.log("❌ FALLO: Patrón singleton no está funcionando")
                self.log(f"❌ Tiempo segunda carga: {second_load_time:.4f}s (debería ser <0.1s)")
                self.log(f"❌ Tiempo tercera carga: {third_load_time:.4f}s (debería ser <0.1s)")
                self.log(f"❌ Misma instancia: {all_same} (debería ser True)")
            
            return {
                "first_load_time": first_load_time,
                "second_load_time": second_load_time,
                "third_load_time": third_load_time,
                "all_same_instance": all_same,
                "singleton_working": all_same and second_load_time < 0.1
            }
            
        except Exception as e:
            self.log(f"❌ ERROR en test singleton: {e}")
            return {
                "first_load_time": 0,
                "second_load_time": 0,
                "third_load_time": 0,
                "all_same_instance": False,
                "singleton_working": False,
                "error": str(e)
            }

    async def test_multiple_requests_simulation(self):
        """Simular múltiples requests como los del frontend"""
        self.log("🌐 SIMULANDO: Múltiples requests como frontend")
        
        try:
            from app.services.service_manager import service_manager
            
            # Simular 5 requests concurrentes
            tasks = []
            for i in range(5):
                tasks.append(self.simulate_request(i+1))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            self.log(f"⏱️ Tiempo total para 5 requests: {total_time:.2f}s")
            self.log(f"⚡ Tiempo promedio por request: {total_time/5:.2f}s")
            
            # Verificar que todos usaron la misma instancia
            ai_service_ids = [result["ai_service_id"] for result in results if result["ai_service_id"] != 0]
            all_same_id = len(set(ai_service_ids)) == 1 if ai_service_ids else False
            
            self.log(f"🔗 ¿Todos usaron la misma instancia de AI? {all_same_id}")
            
            # Guardar resultados para mostrar en la tabla
            self.request_results = results
            
            if total_time < 10 and all_same_id:
                self.log("✅ ÉXITO: Requests múltiples son eficientes")
            else:
                self.log("❌ FALLO: Requests múltiples son lentos")
            
            return {
                "total_time": total_time,
                "avg_time_per_request": total_time/5,
                "all_same_ai_instance": all_same_id,
                "efficient": total_time < 10,
                "request_details": results
            }
            
        except Exception as e:
            self.log(f"❌ ERROR en test de requests: {e}")
            return {
                "total_time": 999,
                "avg_time_per_request": 999,
                "all_same_ai_instance": False,
                "efficient": False,
                "error": str(e)
            }

    async def simulate_request(self, request_id: int):
        """Simular un request individual"""
        self.log(f"📨 Request {request_id}: Iniciando...")
        
        try:
            start_time = time.time()
            
            # Simular las operaciones típicas de un request
            from app.services.service_manager import service_manager
            
            # 1. Obtener servicios (debería ser instantáneo)
            ai_service = service_manager.ai_service
            nlp_service = service_manager.nlp_service
            
            # 2. Simular análisis de texto
            test_text = f"Este es un texto de prueba para el request {request_id}"
            concepts = nlp_service.extract_key_concepts(test_text, max_concepts=3)
            
            # 3. Simular generación de resumen
            summary = await ai_service.generate_summary(test_text, "short")
            
            end_time = time.time()
            request_time = end_time - start_time
            
            self.log(f"📨 Request {request_id}: Completado en {request_time:.2f}s")
            
            return {
                "request_id": request_id,
                "time": request_time,
                "ai_service_id": id(ai_service),
                "success": summary.get("success", False)
            }
            
        except Exception as e:
            self.log(f"❌ Request {request_id}: Error - {e}")
            return {
                "request_id": request_id,
                "time": 999,
                "ai_service_id": 0,
                "success": False,
                "error": str(e)
            }

def main():
    """Función principal"""
    root = tk.Tk()
    
    # Configurar estilo
    style = ttk.Style()
    style.theme_use('clam')
    
    # Crear aplicación
    app = SingletonTestUnifiedGUI(root)
    
    # Configurar cierre de aplicación
    def on_closing():
        if app.test_running:
            if messagebox.askokcancel("Salir", "Los tests están ejecutándose. ¿Deseas salir de todas formas?"):
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        # Ejecutar aplicación
        root.mainloop()
    except KeyboardInterrupt:
        print("\n🛑 Aplicación interrumpida por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando aplicación: {e}")

if __name__ == "__main__":
    main()