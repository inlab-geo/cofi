#
# wigpy/geom.py
#
# Copyright (C) 2015 CSIRO
#
# This file is part of wiglaf
#
# Wiglaf is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# Wiglaf is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with wiglaf.
# If not, see <https://www.gnu.org/licenses/>.
#
# To contact CSIRO about this software you can e-mail
# juerg.hauser@csiro.au
#

import numpy
import numpy.linalg


def get_line_line_intersection(x1,y1,x2,y2,x3,y3,x4,y4):


    #print(x1,y1)
    #print(x2,y2)

    #print(x3,y3)
    #print(x4,y4)

    #print((x1-x2)*(y3-y4))
    #print((y1-y2)*(x3-x4))


    if (((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))==0.0):
        xc=None
        yc=None
        return [xc,yc]

    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    u = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))


    if (t>=0.0 and t<=1.0 and u>=0.0 and u<=1.0):
        xc = x1+t*(x2-x1)
        yc = y1+t*(y2-y1)
    else:
        xc=None
        yc=None

    return [xc,yc]


def get_line_cell_intersection(xa,ya,xb,yb,x0,y0,dx,dy):

    ab=[]

    [xc,yc]=get_line_line_intersection(xa,ya,xb,yb,x0,y0,x0+dx,y0)
    if (xc!=None):
        #print(xc)
        ab.append([xc,yc])
    [xc,yc]=get_line_line_intersection(xa,ya,xb,yb,x0+dx,y0,x0+dx,y0+dy)
    if (xc!=None):
        #print(xc)
        ab.append([xc,yc])
    [xc,yc]=get_line_line_intersection(xa,ya,xb,yb,x0+dx,y0+dy,x0,y0+dy)
    if (xc!=None):
        #print(xc)
        ab.append([xc,yc])
    [xc,yc]=get_line_line_intersection(xa,ya,xb,yb,x0,y0+dy,x0,y0)
    if (xc!=None):
        #print(xc)
        ab.append([xc,yc])
    return ab


def get_line_segments(xa,ya,xb,yb,x0,y0,nx,ny,dx,dy):

    lsg = [];

    for i in range(nx):
        for j in range(ny):
            xc=x0+i*dx
            yc=y0+j*dy
            ab=get_line_cell_intersection(xa,ya,xb,yb,xc,yc,dx,dy)
            #print(ab)
            if ab!=[]:
                ab=numpy.array(ab)
                if numpy.shape(ab)[0]>1:
 	               lsg.append([ab[0,0],ab[0,1],ab[1,0],ab[1,1],i,j])

    return numpy.array(lsg)


def get_track(x0, y0, x1, y1, dd):
    n = int(numpy.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / dd) + 1
    theta = numpy.arctan2(y1 - y0, x1 - x0)
    dx = numpy.cos(theta) * dd
    dy = numpy.sin(theta) * dd
    x = x0
    y = y0
    d = 0.0
    track = []
    track.append(numpy.array([x, y, d]))
    for i in range(1, n+1):
        x = x + dx
        y = y + dy
        d = d + dd
        track.append(numpy.array([x, y, d]))
    return numpy.array(track)


# http://mathworld.wolfram.com/RotationMatrix.html
# Note these are for counter clockwise rotations the clockwise
# rotation is achieved by a negative angle


def get_rotation_matrix_rx(alpha):
    rx = numpy.zeros([3, 3])
    rx[0, 0] = 1.0
    rx[1, 1] = numpy.cos(alpha)
    rx[1, 2] = numpy.sin(alpha)
    rx[2, 1] = - numpy.sin(alpha)
    rx[2, 2] = numpy.cos(alpha)
    return rx


def get_rotation_matrix_ry(beta):
    ry = numpy.zeros([3, 3])
    ry[0, 0] = numpy.cos(beta)
    ry[0, 2] = - numpy.sin(beta)
    ry[1, 1] = 1.0
    ry[2, 0] = numpy.sin(beta)
    ry[2, 2] = numpy.cos(beta)
    return ry


def get_rotation_matrix_rz(gamma):
    rz = numpy.zeros([3, 3])
    rz[0, 0] = numpy.cos(gamma)
    rz[0, 1] = numpy.sin(gamma)
    rz[1, 0] = - numpy.sin(gamma)
    rz[1, 1] = numpy.cos(gamma)
    rz[2, 2] = 1.0
    return rz


def get_plate_corners_from_orientation(
        x, y, z, plngth1, plngth2, pwdth1, pwdth2, thk,
        dipazm, dip, plng):

    Rplng = get_rotation_matrix_rz(plng)
    Rdip = get_rotation_matrix_rx(dip)
    Razm = get_rotation_matrix_rz(dipazm)
    R = numpy.dot(Razm, numpy.dot(Rdip, Rplng))

    # create horizontal plate
    #
    #     7------------6           ---> lngth
    #    /|           /|           /|
    #   4------------5 |      wdth/ | thk
    #   | |          | |            v
    #   | 3------------2
    #   |/           |/
    #   0------------1
    #

    v = numpy.zeros([8, 3])

    v[0, 0] = - plngth1
    v[0, 1] = - pwdth1
    v[0, 2] = - thk / 2.0

    v[1, 0] = + plngth2
    v[1, 1] = - pwdth1
    v[1, 2] = - thk / 2.0

    v[2, 0] = + plngth2
    v[2, 1] = + pwdth2
    v[2, 2] = - thk / 2.0

    v[3, 0] = - plngth1
    v[3, 1] = + pwdth2
    v[3, 2] = - thk / 2.0

    v[4, 0] = - plngth1
    v[4, 1] = - pwdth1
    v[4, 2] = + thk / 2.0

    v[5, 0] = + plngth2
    v[5, 1] = - pwdth1
    v[5, 2] = + thk / 2.0

    v[6, 0] = + plngth2
    v[6, 1] = + pwdth2
    v[6, 2] = + thk / 2.0

    v[7, 0] = - plngth1
    v[7, 1] = + pwdth2
    v[7, 2] = + thk / 2.0

    # apply rotation
    for i in range(8):
        v[i, :] = numpy.dot(R, v[i, :])
        v[i, 0] += x
        v[i, 1] += y
        v[i, 2] += z

    return v


def get_plate_faces_from_orientation(x, y, z, plngth1, plngth2, pwdth1,
                                     pwdth2, thk, dipazm, dip, plng):
    v = get_plate_corners_from_orientation(
            x, y, z, plngth1, plngth2, pwdth1, pwdth2, thk, dipazm, dip, plng)
    f = numpy.zeros([6, 4, 3])

    f[0, 0, :] = v[0, :]
    f[0, 1, :] = v[1, :]
    f[0, 2, :] = v[2, :]
    f[0, 3, :] = v[3, :]

    f[1, 0, :] = v[1, :]
    f[1, 1, :] = v[5, :]
    f[1, 2, :] = v[6, :]
    f[1, 3, :] = v[2, :]

    f[2, 0, :] = v[4, :]
    f[2, 1, :] = v[5, :]
    f[2, 2, :] = v[6, :]
    f[2, 3, :] = v[7, :]

    f[3, 0, :] = v[0, :]
    f[3, 1, :] = v[4, :]
    f[3, 2, :] = v[7, :]
    f[3, 3, :] = v[3, :]

    f[4, 0, :] = v[0, :]
    f[4, 1, :] = v[1, :]
    f[4, 2, :] = v[5, :]
    f[4, 3, :] = v[4, :]

    f[5, 0, :] = v[3, :]
    f[5, 1, :] = v[2, :]
    f[5, 2, :] = v[6, :]
    f[5, 3, :] = v[7, :]

    return f


class Object(object):
    pass


def get_plane_from_points(p0, pp1, pp2):
    plane = Object()
    p1 = pp1-p0
    p2 = pp2-p0

    plane.p0 = p0
    plane.x = numpy.zeros([3])
    plane.x = p1 * 1.0 / numpy.linalg.norm(p1)
    plane.y = numpy.zeros([3])
    plane.y = p2 * 1.0 / numpy.linalg.norm(p2)

    n = numpy.cross(p1, p2)
    plane.n = n * 1.0 / numpy.linalg.norm(n)

    return plane


def get_line_from_points(p0, p1):
    line = Object()
    line.l0 = p0
    line.l = p1 - p0
    return line


def get_cube_edge(v, i):
    if (i == 0):         # bottom
        j0 = 0
        j1 = 1
    elif(i == 1):
        j0 = 1
        j1 = 2
    elif (i == 2):
        j0 = 2
        j1 = 3
    elif (i == 3):
        j0 = 3
        j1 = 0
    elif(i == 4):
        j0 = 4
        j1 = 5
    elif (i == 5):
        j0 = 5
        j1 = 6
    elif (i == 6):
        j0 = 6
        j1 = 7
    elif(i == 7):
        j0 = 7
        j1 = 4
    elif(i == 8):    # sides
        j0 = 0
        j1 = 4
    elif (i == 9):
        j0 = 1
        j1 = 5
    elif(i == 10):
        j0 = 2
        j1 = 6
    elif (i == 11):
        j0 = 3
        j1 = 7

    edge = get_line_from_points(v[j0, :], v[j1, :])

    return edge


def get_line_plane_intersection(plane, line, iedge):
    p = numpy.zeros([3])
    v = plane.p0 - line.l0
    num = numpy.dot(v, plane.n)
    denom = numpy.dot(line.l, plane.n)

    eps = 1.0e-12
    if (numpy.abs(denom) < eps):
        ifail = 1
    else:
        d = num / denom
        ifail = 1

        if(iedge == 0 and d >= 0.0 - eps and d < 1.0):
            ifail = 0
        elif (iedge == 1 and d >= 0.0 - eps and d < 1.0):
            ifail = 0
        elif (iedge == 2 and d > 0.0 and d <= 1.0 + eps):
            ifail = 0
        elif (iedge == 3 and d >= 0.0 - eps and d < 1.0):
            ifail = 0
        elif (iedge == 4 and d >= 0.0 - eps and d < 1.0):
            ifail = 0
        elif (iedge == 5 and d >= 0.0 - eps and d < 1.0):
            ifail = 0
        elif (iedge == 6 and d > 0.0 and d <= 1.0 + eps):
            ifail = 0
        elif (iedge == 7 and d >= 0.0 - eps and d < 1.0):
            ifail = 0
        elif (iedge == 8 and d > 0.0 and d < 1.0):
            ifail = 0
        elif (iedge == 9 and d > 0.0 and d < 1.0):
            ifail = 0
        elif(iedge == 10 and d > 0.0 and d < 1.0):
            ifail = 0
        elif (iedge == 11 and d > 0.0 and d < 1.0):
            ifail = 0

        if (ifail == 0):
            p = line.l0 + line.l * d
        else:
            ifail = 1

    return ifail, p


def clockwise_angle_between_vectors_in_plane(plane, v1, v2):
    dot = numpy.dot(v1, v2)
    det = (v1[0] * v2[1] * plane.n[2] + v2[0] * plane.n[1] * v1[2] +
           plane.n[0] * v1[1] * v2[2] - v1[2] * v2[1] * plane.n[0] -
           v2[2] * plane.n[1] * v1[0] - plane.n[2] * v1[1] * v2[0])
    theta = numpy.arctan2(det, dot)
    return theta


def get_plane_plate_intersection(
        p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, x, y, z,
        plngth1, plngth2, pwdth1, pwdth2, thk, dipazm, dip, plng):
    plane = get_plane_from_points(
            numpy.array([p0x, p0y, p0z]), numpy.array([p1x, p1y, p1z]),
            numpy.array([p2x, p2y, p2z]))
    v = get_plate_corners_from_orientation(
            x, y, z, plngth1, plngth2, pwdth1, pwdth2, thk, dipazm, dip, plng)

    pp = []
    for i in range(12):
        edge = get_cube_edge(v, i)
        ifail, p = get_line_plane_intersection(plane, edge, i)
        if (ifail == 0):
            pp.append(p)

    pp = numpy.array(pp)
    pc = numpy.mean(pp, axis=0)
    theta = []
    v0 = pp[0, :] - pc
    for i in range(numpy.shape(pp)[0]):
        v = pp[i, :] - pc
        theta.append(clockwise_angle_between_vectors_in_plane(plane, v, v0))
    idx = numpy.argsort(theta)

    ply = []
    for i in range(len(idx)):
        ply.append(pp[idx[i], :])

    return numpy.array(ply)
