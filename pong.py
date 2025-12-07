import math

import pygame


class RigidBody:
    def __init__(self, x, y, vertices, mass=1.0, is_static=False):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = 0.0
        self.ang_vel = 0.0

        # 로컬 좌표계 정점
        self.vertices_local = [pygame.math.Vector2(v) for v in vertices]
        self.vertices_world = []
        self.mass = mass

        # 정적 물체거나 질량이 0이면 무한 질량 처리
        if is_static or mass == 0:
            self.inv_mass = 0.0
            self.inv_inertia = 0.0
            self.is_static = True
        else:
            self.inv_mass = 1.0 / mass
            self.inertia = self.calculate_inertia()
            self.inv_inertia = 1.0 / self.inertia if self.inertia > 0 else 0.0
            self.is_static = False

        self.restitution = 0.8  # 반발 계수
        self.friction = 0.1  # 마찰 계수

        self.update_vertices()

    def calculate_inertia(self):
        # 관성 모멘트 근사 계산
        if self.mass == 0:
            return 0
        radius_sq = max([v.length_squared() for v in self.vertices_local])
        return 0.5 * self.mass * radius_sq

    def update_vertices(self):
        # 로컬 정점을 월드 좌표로 변환 (회전 적용)
        self.vertices_world = []
        c = math.cos(self.angle)
        s = math.sin(self.angle)
        for v in self.vertices_local:
            rx = v.x * c - v.y * s
            ry = v.x * s + v.y * c
            self.vertices_world.append(
                pygame.math.Vector2(self.pos.x + rx, self.pos.y + ry)
            )

    def integrate(self, dt):
        if self.is_static:
            return

        self.pos += self.vel * dt
        self.angle += self.ang_vel * dt
        self.update_vertices()


# --- SAT (Separating Axis Theorem) 충돌 감지 ---
def check_collision_sat(body_a, body_b):
    overlap = float("inf")
    normal = pygame.math.Vector2(0, 0)

    bodies = [body_a, body_b]
    for i in range(2):
        poly = bodies[i]

        for j in range(len(poly.vertices_world)):
            p1 = poly.vertices_world[j]
            p2 = poly.vertices_world[(j + 1) % len(poly.vertices_world)]

            edge = p2 - p1
            axis = pygame.math.Vector2(-edge.y, edge.x).normalize()

            min_a, max_a = project_vertices(body_a.vertices_world, axis)
            min_b, max_b = project_vertices(body_b.vertices_world, axis)

            if min_a >= max_b or min_b >= max_a:
                return False, None, None

            axis_overlap = min(max_a, max_b) - max(min_a, min_b)
            if axis_overlap < overlap:
                overlap = axis_overlap
                normal = axis

    direction = body_b.pos - body_a.pos
    if direction.dot(normal) < 0:
        normal = -normal

    return True, normal, overlap


def project_vertices(vertices, axis):
    min_proj = float("inf")
    max_proj = float("-inf")
    for v in vertices:
        proj = v.dot(axis)
        if proj < min_proj:
            min_proj = proj
        if proj > max_proj:
            max_proj = proj
    return min_proj, max_proj


# --- 충돌 해결 (Impulse Response) ---
def resolve_collision(body_a, body_b, normal, depth):
    # 1. 위치 보정 (Penetration Resolution)
    percent = 0.2
    slop = 0.01
    correction = (
        normal * max(depth - slop, 0.0) / (body_a.inv_mass + body_b.inv_mass) * percent
    )

    if not body_a.is_static:
        body_a.pos -= correction * body_a.inv_mass
    if not body_b.is_static:
        body_b.pos += correction * body_b.inv_mass

    # 2. 속도 해결 (Restitution)
    rv = body_b.vel - body_a.vel
    vel_along_normal = rv.dot(normal)

    if vel_along_normal > 0:
        return

    e = min(body_a.restitution, body_b.restitution)
    j = -(1 + e) * vel_along_normal
    j /= body_a.inv_mass + body_b.inv_mass

    impulse = normal * j

    body_a.vel -= impulse * body_a.inv_mass
    body_b.vel += impulse * body_b.inv_mass

    # 3. 마찰 및 회전 (Friction & Angular Impulse)
    tangent = rv - (normal * rv.dot(normal))
    if tangent.length_squared() > 0.0001:
        tangent = tangent.normalize()

    jt = -rv.dot(tangent)
    jt /= body_a.inv_mass + body_b.inv_mass

    mu = (body_a.friction + body_b.friction) * 0.5
    friction_impulse = tangent * jt * mu

    body_a.vel -= friction_impulse * body_a.inv_mass
    body_b.vel += friction_impulse * body_b.inv_mass

    torque_factor = 0.1
    if not body_a.is_static:
        body_a.ang_vel -= jt * torque_factor * body_a.inv_inertia
    if not body_b.is_static:
        body_b.ang_vel += jt * torque_factor * body_b.inv_inertia


# ==========================================
# [Game Logic]
# ==========================================


def create_rect_vertices(w, h):
    return [
        pygame.math.Vector2(-w / 2, -h / 2),
        pygame.math.Vector2(w / 2, -h / 2),
        pygame.math.Vector2(w / 2, h / 2),
        pygame.math.Vector2(-w / 2, h / 2),
    ]


def create_hexagon_vertices(radius):
    verts = []
    for i in range(6):
        angle = math.radians(60 * i)
        verts.append(
            pygame.math.Vector2(radius * math.cos(angle), radius * math.sin(angle))
        )
    return verts


def main():
    # Font 모듈 호환성 이슈 방지
    pygame.init()
    if pygame.font:
        pygame.font = None

    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Project #3: Rigid Body Pong (Fixed)")
    clock = pygame.time.Clock()

    # --- 객체 생성 ---
    # Ball: Dynamic Body
    ball = RigidBody(400, 300, create_hexagon_vertices(15), mass=1.0, is_static=False)
    ball.vel = pygame.math.Vector2(300, 100)

    # Paddle: Kinematic Body (위치를 직접 제어하므로 질량 0/Static 처리)
    paddle_w, paddle_h = 20, 100
    player = RigidBody(
        50, 300, create_rect_vertices(paddle_w, paddle_h), mass=0, is_static=True
    )
    ai = RigidBody(
        750, 300, create_rect_vertices(paddle_w, paddle_h), mass=0, is_static=True
    )

    # Walls: Static Body
    wall_top = RigidBody(
        400, -10, create_rect_vertices(800, 20), mass=0, is_static=True
    )
    wall_bottom = RigidBody(
        400, 610, create_rect_vertices(800, 20), mass=0, is_static=True
    )

    bodies = [ball, player, ai, wall_top, wall_bottom]
    score = [0, 0]

    running = True
    print("Game Started! Check console for score.")

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Player Input (Kinematic Movement) ---
        keys = pygame.key.get_pressed()
        player_speed = 400

        if keys[pygame.K_UP]:
            player.pos.y -= player_speed * dt
        if keys[pygame.K_DOWN]:
            player.pos.y += player_speed * dt

        # --- AI Logic ---
        if ball.pos.y > ai.pos.y + 10:
            ai.pos.y += 250 * dt
        elif ball.pos.y < ai.pos.y - 10:
            ai.pos.y -= 250 * dt

        # --- Boundary Check & Vertex Update ---
        for p in [player, ai]:
            if p.pos.y < 50:
                p.pos.y = 50
            if p.pos.y > 550:
                p.pos.y = 550
            # Kinematic 이동 후에는 정점 업데이트 필수
            p.update_vertices()

        # --- Physics Update ---
        for body in bodies:
            body.integrate(dt)

        for other in [player, ai, wall_top, wall_bottom]:
            is_colliding, normal, depth = check_collision_sat(ball, other)
            if is_colliding:
                resolve_collision(ball, other, normal, depth)

        # --- Scoring ---
        if ball.pos.x < 0:
            score[1] += 1
            print(f"Goal! Player: {score[0]} | AI: {score[1]}")
            ball.pos = pygame.math.Vector2(400, 300)
            ball.vel = pygame.math.Vector2(200, 100)
            ball.ang_vel = 0
            ball.update_vertices()
        elif ball.pos.x > width:
            score[0] += 1
            print(f"Goal! Player: {score[0]} | AI: {score[1]}")
            ball.pos = pygame.math.Vector2(400, 300)
            ball.vel = pygame.math.Vector2(-200, 100)
            ball.ang_vel = 0
            ball.update_vertices()

        # --- Rendering ---
        screen.fill((30, 30, 30))

        for body in bodies:
            points = [(p.x, p.y) for p in body.vertices_world]

            if body == ball:
                color = (200, 50, 50)
            elif body.is_static and body not in [wall_top, wall_bottom]:
                color = (50, 200, 50)
            else:
                color = (100, 100, 255)

            pygame.draw.polygon(screen, color, points)
            pygame.draw.lines(screen, (255, 255, 255), True, points, 2)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
