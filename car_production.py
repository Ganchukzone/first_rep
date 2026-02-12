import tkinter as tk
from tkinter import ttk, messagebox
import psycopg2
from datetime import datetime
import hashlib

class DatabaseApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Система управления автомобильным производством")
        self.root.geometry("500x400")
        
        # Параметры подключения к БД
        self.db_params = {
            'host': 'localhost',
            'database': 'car_production',
            'user': 'postgres',
            'password': 'password',
            'port': '5432'
        }
        
        # Текущий пользователь и его роль
        self.current_user = None
        self.current_role = None
        
        self.create_login_interface()
    
    def create_connection(self):
        """Создание подключения к базе данных"""
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось подключиться к БД: {e}")
            return None
    
    def hash_password(self, password):
        """Хэширование пароля"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_login_interface(self):
        """Создание интерфейса авторизации"""
        # Очистка окна
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.root.title("Авторизация")
        self.root.geometry("400x300")
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Авторизация", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=20)
        
        # Поля для ввода
        ttk.Label(main_frame, text="Логин:").pack(pady=(10, 0))
        self.login_entry = ttk.Entry(main_frame, width=30)
        self.login_entry.pack()
        
        ttk.Label(main_frame, text="Пароль:").pack(pady=(10, 0))
        self.password_entry = ttk.Entry(main_frame, width=30, show="*")
        self.password_entry.pack()
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Войти", 
                  command=self.authenticate_user).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Выйти", 
                  command=self.root.quit).pack(side=tk.LEFT, padx=5)
    
    def authenticate_user(self):
        """Аутентификация пользователя"""
        login = self.login_entry.get()
        password = self.password_entry.get()
        
        if not login or not password:
            messagebox.showwarning("Ошибка", "Введите логин и пароль")
            return
        
        hashed_password = self.hash_password(password)
        
        conn = self.create_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                # Проверка пользователя
                cursor.execute("""
                    SELECT u.id, u.login, r.role_name 
                    FROM users u 
                    JOIN roles r ON u.role_id = r.id 
                    WHERE u.login = %s AND u.password_hash = %s
                """, (login, hashed_password))
                
                user_data = cursor.fetchone()
                
                if user_data:
                    self.current_user = {
                        'id': user_data[0],
                        'login': user_data[1],
                        'role': user_data[2]
                    }
                    self.current_role = user_data[2]
                    
                    messagebox.showinfo("Успех", f"Добро пожаловать, {login}!")
                    self.create_main_interface()
                else:
                    messagebox.showerror("Ошибка", "Неверный логин или пароль")
                
                cursor.close()
                conn.close()
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка аутентификации: {e}")
    
    def check_permission(self, required_permission):
        """Проверка прав доступа"""
        conn = self.create_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM role_permissions rp
                    JOIN permissions p ON rp.permission_id = p.id
                    JOIN roles r ON rp.role_id = r.id
                    WHERE r.role_name = %s AND p.permission_name = %s
                """, (self.current_role, required_permission))
                
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                return count > 0
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка проверки прав: {e}")
                return False
        return False
    
    def create_main_interface(self):
        """Создание главного интерфейса после авторизации"""
        # Очистка окна
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.root.title(f"Система управления - {self.current_user['login']} ({self.current_role})")
        self.root.geometry("600x400")
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок с информацией о пользователе
        title_text = f"Система управления автомобильным производством\n"
        title_text += f"Пользователь: {self.current_user['login']} | Роль: {self.current_role}"
        
        title_label = ttk.Label(main_frame, text=title_text, 
                               font=('Arial', 12), justify=tk.CENTER)
        title_label.pack(pady=20)
        
        # Кнопки для разных ролей
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Кнопки доступные всем ролям
        if self.check_permission('view_production'):
            ttk.Button(button_frame, text="Производство", 
                      command=self.open_production, width=20).pack(pady=5)
        
        if self.check_permission('view_sales'):
            ttk.Button(button_frame, text="Реализация", 
                      command=self.open_sales, width=20).pack(pady=5)
        
        if self.check_permission('view_models'):
            ttk.Button(button_frame, text="Модели", 
                      command=self.open_models, width=20).pack(pady=5)
        
        # Кнопки для администраторов и менеджеров
        if self.check_permission('manage_employees'):
            ttk.Button(button_frame, text="Сотрудники", 
                      command=self.open_employees, width=20).pack(pady=5)
        
        if self.check_permission('manage_components'):
            ttk.Button(button_frame, text="Комплектующие", 
                      command=self.open_components, width=20).pack(pady=5)
        
        if self.check_permission('manage_supplies'):
            ttk.Button(button_frame, text="Поставки", 
                      command=self.open_supplies, width=20).pack(pady=5)
        
        if self.check_permission('manage_car_types'):
            ttk.Button(button_frame, text="Типы автомобилей", 
                      command=self.open_car_types, width=20).pack(pady=5)
        
        # Кнопка выхода
        ttk.Button(main_frame, text="Выйти из системы", 
                  command=self.logout, width=20).pack(pady=20)
    
    def logout(self):
        """Выход из системы"""
        self.current_user = None
        self.current_role = None
        self.create_login_interface()
    
    def open_table_window(self, title, table_name, columns, read_only=False):
        """Открытие окна для работы с таблицей"""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("800x600")
        
        # Фрейм для таблицы
        table_frame = ttk.Frame(window)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Дерево для отображения данных
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Настройка заголовков
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Фрейм для кнопок
        button_frame = ttk.Frame(window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        if not read_only:
            if self.check_permission(f'add_{table_name}'):
                ttk.Button(button_frame, text="Добавить", 
                          command=lambda: self.add_record(window, table_name, columns, tree)).pack(side=tk.LEFT, padx=5)
            
            if self.check_permission(f'edit_{table_name}'):
                ttk.Button(button_frame, text="Изменить выбранное", 
                          command=lambda: self.edit_record(window, table_name, columns, tree)).pack(side=tk.LEFT, padx=5)
            
            if self.check_permission(f'delete_{table_name}'):
                ttk.Button(button_frame, text="Удалить выбранное", 
                          command=lambda: self.delete_record(table_name, tree)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Обновить", 
                  command=lambda: self.load_table_data(tree, table_name)).pack(side=tk.LEFT, padx=5)
        
        # Загрузка данных
        self.load_table_data(tree, table_name)
        
        return window, tree
    
    def load_table_data(self, tree, table_name):
        """Загрузка данных в таблицу"""
        # Очистка таблицы
        for item in tree.get_children():
            tree.delete(item)
        
        conn = self.create_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                
                for row in rows:
                    tree.insert("", tk.END, values=row)
                
                cursor.close()
                conn.close()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки данных: {e}")
    
    def add_record(self, parent_window, table_name, columns, tree):
        """Добавление новой записи"""
        self.show_edit_dialog(parent_window, table_name, columns, tree, None)
    
    def edit_record(self, parent_window, table_name, columns, tree):
        """Редактирование выбранной записи"""
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите запись для редактирования")
            return
        
        item = tree.item(selected[0])
        self.show_edit_dialog(parent_window, table_name, columns, tree, item['values'])
    
    def delete_record(self, table_name, tree):
        """Удаление выбранной записи"""
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите запись для удаления")
            return
        
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить выбранную запись?"):
            item = tree.item(selected[0])
            record_id = item['values'][0]
            
            conn = self.create_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"DELETE FROM {table_name} WHERE id = %s", (record_id,))
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    self.load_table_data(tree, table_name)
                    messagebox.showinfo("Успех", "Запись удалена")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка удаления: {e}")
    
    def show_edit_dialog(self, parent_window, table_name, columns, tree, values):
        """Диалоговое окно для добавления/редактирования записи"""
        dialog = tk.Toplevel(parent_window)
        dialog.title("Редактирование записи" if values else "Добавление записи")
        dialog.geometry("400x300")
        dialog.transient(parent_window)
        dialog.grab_set()
        
        entries = {}
        
        # Создание полей ввода
        for i, column in enumerate(columns):
            ttk.Label(dialog, text=column).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            entry = ttk.Entry(dialog, width=30)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky=tk.W)
            
            if values and i < len(values):
                entry.insert(0, str(values[i]))
            
            entries[column] = entry
        
        def save_changes():
            try:
                conn = self.create_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # Подготовка данных
                    data = [entry.get() for entry in entries.values()]
                    
                    if values:  # Редактирование
                        placeholders = ", ".join([f"{col} = %s" for col in columns])
                        query = f"UPDATE {table_name} SET {placeholders} WHERE id = %s"
                        cursor.execute(query, data + [values[0]])
                    else:  # Добавление
                        placeholders = ", ".join(["%s"] * len(columns))
                        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                        cursor.execute(query, data)
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    self.load_table_data(tree, table_name)
                    dialog.destroy()
                    messagebox.showinfo("Успех", "Данные сохранены")
                    
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения: {e}")
        
        ttk.Button(dialog, text="Сохранить", command=save_changes).grid(
            row=len(columns), column=0, columnspan=2, pady=20)
    
    # Методы для открытия конкретных таблиц
    def open_employees(self):
        self.open_table_window("Сотрудники", "employees", 
                              ["id", "name", "position", "department", "hire_date", "salary"])
    
    def open_production(self):
        self.open_table_window("Производство", "production", 
                              ["id", "model_id", "employee_id", "production_date", "quantity", "status"])
    
    def open_components(self):
        self.open_table_window("Комплектующие", "components", 
                              ["id", "name", "type", "supplier", "price", "quantity_in_stock"])
    
    def open_sales(self):
        self.open_table_window("Реализация", "sales", 
                              ["id", "model_id", "sale_date", "quantity", "price", "customer"])
    
    def open_models(self):
        self.open_table_window("Модели", "models", 
                              ["id", "name", "car_type_id", "engine_power", "price", "production_year"])
    
    def open_supplies(self):
        self.open_table_window("Поставки", "supplies", 
                              ["id", "component_id", "supplier", "supply_date", "quantity", "price"])
    
    def open_car_types(self):
        self.open_table_window("Типы автомобилей", "car_types", 
                              ["id", "type_name", "description", "category"])

    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

def main():
    app = DatabaseApp()
    app.run()

if __name__ == "__main__":
    main()