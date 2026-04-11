from api.routing import AmbulanceRouter

r = AmbulanceRouter()
result = r.route(28.5439, 77.3305)

print("Routing result:")
for k, v in result.items():
    if k != "waypoints":
        print(f"  {k}: {v}")
print(f"  waypoints: {len(result['waypoints'])} nodes")
print()
print("Waypoint path:")
for w in result["waypoints"]:
    print(f"  -> {w['name']}")
